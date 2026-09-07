"""FreeDehaze research engine, extracted from libs/model/sd_pipeline.py.

Branch order: negative dehazing, negative perception, reconstruction,
conditional dehazing, conditional perception.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import kornia
import numpy as np
import omegaconf
import ot
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
from diffusers.utils import BaseOutput, deprecate
from image_utils import PILtoTensor
from attention import register_time
from priors import (
    DarkChannel,
    adaptive_instance_normalization,
    modified_sigmoid_adjusted,
    compute_mean_cov,
)
from conv import prep_unet_conv
from attention import prep_unet_attention
from image_utils import _in_step, _classify_blocks


class StableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


class FreeControlSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt2=None,
        lq_img=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        config: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        inverted_data=None,
    ):
        assert config is not None, "config is required for FreeControl pipeline"
        self.input_config = config
        self.DCP_prior = (
            torch.tensor(
                DarkChannel(
                    np.array(lq_img.resize(self.img_size)),
                    config.guidance.others.DCP_kernal,
                    config.guidance.others.sky,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        qk_inject = config.guidance.cross_attn.qk_inject
        conv_inject = config.guidance.cross_attn.conv_inject
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        self.qk_injection_timesteps = self.scheduler.timesteps[
            : int(len(self.scheduler.timesteps) * qk_inject)
        ]
        self.conv_injection_timesteps = self.scheduler.timesteps[
            : int(len(self.scheduler.timesteps) * conv_inject)
        ]
        self.unet = prep_unet_attention(
            self.unet, config, self.qk_injection_timesteps, DCP_mask=self.DCP_prior
        )
        self.unet = prep_unet_conv(self.unet, self.conv_injection_timesteps)
        self.load_pca_info()
        self.running_device = self._execution_device
        self.ref_mask_record = None
        height = (
            self.img_size[1] or self.unet.config.sample_size * self.vae_scale_factor
        )
        width = self.img_size[0] or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self.cross_attn_probs: Dict = {"channels": 0, "probs": None}
        self.cross_attn_probs_all = {}
        self.cross_attn_channel_all = {}
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        if config.guidance.others.longclip:
            prompt_embeds = self.l_text_encoder(
                self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=248,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
            )[0]
            negative_prompt_embeds = self.l_text_encoder(
                self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=248,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
            )[0]
        else:
            (prompt_embeds, negative_prompt_embeds) = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
        if prompt2 is not None:
            prompt_embeds2 = self.l_text_encoder(
                self.tokenizer(
                    prompt2,
                    padding="max_length",
                    max_length=248,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
            )[0]
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        same_latent = config.sd_config.same_latent
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if same_latent:
            keep_latents = latents
        latents = torch.cat([latents] * 2, dim=0)
        if inverted_data is not None:
            num_example_sample: int = len(inverted_data["condition_input"])
        else:
            num_example_sample = 1
        num_appearance_sample: int = 0
        num_control_samples: int = batch_size * num_images_per_prompt
        if num_appearance_sample == 0:
            num_appearance_sample = num_control_samples
        total_samples: int = 0
        if config.data.inversion.method == "DDIM":
            uncond_example_ids: List[int] = list()
            total_samples += (
                2 * (num_control_samples + num_appearance_sample) + num_example_sample
            )
        else:
            uncond_example_ids: List[int] = np.arange(num_example_sample).tolist()
            total_samples += 2 * (
                num_control_samples + num_appearance_sample + num_example_sample
            )
        cond_example_ids: List[int] = (
            np.arange(0, num_example_sample, 1)
            + (num_control_samples * 2 + len(uncond_example_ids))
        ).tolist()
        cond_control_ids: List[int] = (
            np.arange(0, num_control_samples, 1) + (cond_example_ids[-1] + 1)
        ).tolist()
        cond_appearance_ids: List[int] = (
            np.arange(0, num_control_samples, 1) + (cond_control_ids[-1] + 1)
        ).tolist()
        example_ids = uncond_example_ids + cond_example_ids
        keep_ids: List[int] = [
            ids for ids in np.arange(total_samples).tolist() if ids not in example_ids
        ]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.guidance_config = config.guidance
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if config.guidance.others.blur_guidance:
            lq_img = self.image_processor.preprocess(lq_img.resize(self.img_size))
            anchor_latents = self.prepare_image_latents(
                lq_img, 1, self.vae.dtype, self.device, mean=True
            )
        self.mask_degrade_ema = None
        self.mask_degrade_ema_ori = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.t = t
                register_time(self, t.item())
                score = None
                assert do_classifier_free_guidance, (
                    "Currently only support classifier free guidance"
                )
                if inverted_data is not None:
                    step_timestep: int = t.detach().cpu().item()
                    assert (
                        step_timestep
                        in inverted_data["condition_input"][0]["all_latents"].keys()
                    ), f"timestep {step_timestep} not in inverse samples keys"
                    data_samples_latent: torch.Tensor = inverted_data[
                        "condition_input"
                    ][0]["all_latents"][step_timestep]
                    data_samples_latent = data_samples_latent.to(
                        device=self.running_device, dtype=prompt_embeds.dtype
                    )
                    if config.data.inversion.method == "DDIM":
                        if (
                            i == 0
                            and same_latent
                            and config.sd_config.appearnace_same_latent
                        ):
                            latents = data_samples_latent.repeat(2, 1, 1, 1)
                        elif (
                            i == 0
                            and same_latent
                            and (not config.sd_config.appearnace_same_latent)
                        ):
                            latents = torch.cat(
                                [data_samples_latent, keep_latents], dim=0
                            )
                            print("Latents shape", latents.shape)
                        if i > config.guidance.others.exam_fusion:
                            weight = F.interpolate(
                                self.DCP_prior, latents[0:1].shape[2:]
                            ).half()
                            sky_index = weight == 0.1
                            torch.mean(weight[~sky_index])
                            torch.std(weight[~sky_index])
                            data_samples = adaptive_instance_normalization(
                                data_samples_latent, latents[0:1]
                            )
                            latents_exam = (
                                weight * latents[0:1] + (1 - weight) * data_samples
                            )
                            latents_exam = (
                                latents_exam * ~sky_index
                                + data_samples_latent * sky_index
                            )
                            latent_list: List[torch.Tensor] = [
                                latents,
                                latents_exam,
                                latents,
                            ]
                        else:
                            latent_list: List[torch.Tensor] = [
                                latents,
                                data_samples_latent,
                                latents,
                            ]
                            latents_exam = data_samples_latent
                    else:
                        raise NotImplementedError("Currently only support DDIM method")
                else:
                    if i == 0:
                        latents_exam = keep_latents
                    latent_list: List[torch.Tensor] = [latents, latents_exam, latents]
                latent_model_input: torch.Tensor = torch.cat(latent_list, dim=0).to(
                    self._execution_device
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                ).detach()
                if inverted_data is not None:
                    if config.data.inversion.method == "DDIM":
                        if config.guidance.others.longclip:
                            ref_prompt_embeds = inverted_data["condition_input"][0][
                                "prompt_embeds"
                            ].to(self._execution_device)
                        else:
                            ref_prompt_embeds = prompt_embeds.chunk(2)[0]
                        if t.item() > config.guidance.others.dehaze_CFG:
                            step_prompt_embeds_list: List[torch.Tensor] = (
                                [prompt_embeds.chunk(2)[0]] * 2
                                + [ref_prompt_embeds]
                                + [ref_prompt_embeds]
                                + [prompt_embeds.chunk(2)[1]]
                            )
                        else:
                            step_prompt_embeds_list: List[torch.Tensor] = (
                                [prompt_embeds.chunk(2)[0]] * 2
                                + [ref_prompt_embeds]
                                + [prompt_embeds.chunk(2)[1]] * 2
                            )
                    else:
                        raise NotImplementedError("Currently only support DDIM method")
                else:
                    step_prompt_embeds_list: List[torch.Tensor] = (
                        [prompt_embeds.chunk(2)[0]] * 2
                        + [prompt_embeds2]
                        + [prompt_embeds.chunk(2)[1]] * 2
                    )
                step_prompt_embeds = torch.cat(step_prompt_embeds_list, dim=0).to(
                    self._execution_device
                )
                require_grad_flag = False
                if _in_step(self.guidance_config.pca_guidance, i):
                    require_grad_flag = True
                if require_grad_flag:
                    latent_model_input.requires_grad_(True)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=step_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=step_prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                loss = 0
                if _in_step(self.guidance_config.pca_guidance, i):
                    try:
                        select_feature = (
                            self.guidance_config.pca_guidance.select_feature
                        )
                    except (AttributeError, KeyError):
                        select_feature = "key"
                    if (
                        select_feature == "query"
                        or select_feature == "key"
                        or select_feature == "value"
                    ):
                        pca_loss = self.compute_attn_pca_loss(
                            cond_control_ids,
                            cond_example_ids,
                            cond_appearance_ids,
                            i,
                            config,
                        )
                        loss += pca_loss
                    elif select_feature == "conv":
                        pca_loss = self.compute_conv_pca_loss(
                            cond_control_ids, cond_example_ids, cond_appearance_ids, i
                        )
                        loss += pca_loss
                temp_control_ids = None
                if do_classifier_free_guidance:
                    noise_exam = noise_pred[2]
                    noise_pred = noise_pred[keep_ids]
                    (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
                    cur_guidance = max(4, guidance_scale * 2 * i / len(timesteps))
                    cur_guidance2 = max(1, 4 * 2 * i / len(timesteps))
                    noise_pred = noise_pred_uncond + cur_guidance * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if t.item() > config.guidance.others.dehaze_CFG:
                        noise_pred[0] = noise_pred_text[0]
                    else:
                        noise_pred[0] = noise_pred_uncond[0] + cur_guidance2 * (
                            noise_pred_text[0] - noise_pred_uncond[0]
                        )
                if isinstance(loss, torch.Tensor):
                    if loss != 0:
                        gradient = torch.autograd.grad(
                            loss, latent_model_input, allow_unused=False
                        )[0]
                    else:
                        gradient = torch.zeros_like(latent_model_input)
                    gradient = gradient[cond_control_ids]
                    if config.guidance.others.grad_clip:
                        max_norm = 1.0
                        total_norm = torch.sum(gradient.detach().norm(2))
                        total_norm = total_norm**0.5
                        clip_coef = max_norm / (total_norm + 1e-06)
                        if clip_coef < 1:
                            gradient *= clip_coef
                    assert gradient is not None, f"Step {i}: grad is None"
                    score = gradient.detach()
                    temp_control_ids: List[int] = np.arange(
                        num_control_samples
                    ).tolist()
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    score=score,
                    guidance_scale=self.input_config.sd_config.grad_guidance_scale,
                    indices=temp_control_ids,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0].detach()
                if inverted_data is not None:
                    latents_exam = self.scheduler.step(
                        noise_exam,
                        t,
                        data_samples_latent,
                        score=None,
                        guidance_scale=self.input_config.sd_config.grad_guidance_scale,
                        indices=temp_control_ids,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0].detach()
                else:
                    latents_exam = self.scheduler.step(
                        noise_exam,
                        t,
                        latents_exam,
                        score=None,
                        guidance_scale=self.input_config.sd_config.grad_guidance_scale,
                        indices=temp_control_ids,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0].detach()
                if (
                    config.guidance.others.blur_guidance
                    and t
                    <= timesteps[-int(np.ceil(len(timesteps) * (1 - qk_inject) * 0.1))]
                ):
                    noise = torch.randn_like(anchor_latents)
                    anchor_latents_noise = self.scheduler.add_noise(
                        anchor_latents, noise, t
                    )
                    latents_blur = anchor_latents_noise - latents[0:1]
                    latents_blur = kornia.filters.gaussian_blur2d(
                        latents_blur, (3, 3), (0.5, 0.5)
                    )
                    latents[0:1] = anchor_latents_noise - latents_blur
                if i == len(timesteps) - 1 or (
                    i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == "latent":
            with torch.no_grad():
                latents = torch.cat([latents, latents_exam], dim=0)
                image = self.vae.decode(
                    (latents / self.vae.config.scaling_factor).to(self.vae.dtype),
                    return_dict=False,
                )[0]
                if not torch.isfinite(image).all():
                    raise RuntimeError("Decoded output contains NaN/Inf")
                has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def load_pca_info(self):
        path = self.input_config.sd_config.pca_paths[0]
        if getattr(self, "_pca_path", None) != path:
            self.loaded_pca_info = torch.load(
                path, map_location="cpu", weights_only=True
            )
            self._pca_path = path

    def _compute_feat_loss(
        self,
        feat,
        pca_info,
        cond_control_ids,
        cond_example_ids,
        cond_appearance_ids,
        step,
        config,
        reg_included=False,
        reg_feature=None,
    ):
        loss: List[torch.Tensor] = []
        # PCA projection/loss in float32 avoids half precision covariance overflow.
        feat = feat.float()
        feat_mean: torch.Tensor = pca_info["mean"].to(
            self.running_device, dtype=torch.float32
        )
        feat_basis: torch.Tensor = pca_info["basis"].to(
            self.running_device, dtype=torch.float32
        )
        n_components: int = (
            self.guidance_config.pca_guidance.structure_guidance.n_components
        )
        centered_feat = feat - feat_mean
        feat_proj = torch.matmul(centered_feat, feat_basis.T)
        if self.guidance_config.pca_guidance.structure_guidance.normalized:
            feat_proj = feat_proj.permute(0, 2, 1)
            feat_proj_max = feat_proj.max(dim=-1, keepdim=True)[0].detach()
            feat_proj_min = feat_proj.min(dim=-1, keepdim=True)[0].detach()
            feat_proj = (feat_proj - feat_proj_min) / (
                feat_proj_max - feat_proj_min + 1e-07
            )
            feat_proj = feat_proj.permute(0, 2, 1)
        feat_proj = feat_proj[:, :, :n_components]
        if self.guidance_config.pca_guidance.structure_guidance.mask_tr > 0:
            with torch.no_grad():
                index = (
                    torch.mean(feat_proj[cond_control_ids][0], dim=0)
                    - torch.mean(feat_proj[cond_example_ids][0], dim=0)
                    < 0
                )
                index2 = (
                    torch.mean(feat_proj[cond_appearance_ids][0], dim=0)
                    - torch.mean(feat_proj[cond_example_ids][0], dim=0)
                    < 0
                )
                torch.mean(feat_proj[cond_appearance_ids][0], dim=0) - torch.mean(
                    feat_proj[cond_control_ids][0], dim=0
                ) < 0
                index = index2 * index
                scale_down = int(
                    np.sqrt(
                        self.DCP_prior.shape[2]
                        * self.DCP_prior.shape[3]
                        / feat.shape[1]
                    )
                )
                new_w = int(self.DCP_prior.shape[2] / scale_down)
                new_h = int(self.DCP_prior.shape[3] / scale_down)
                if config.guidance.others.loss_DCP_weight == 0:
                    weight = torch.ones_like(self.DCP_prior)
                    weight_resize = F.interpolate(
                        self.DCP_prior, (new_w, new_h)
                    ).reshape(-1, 1)
                else:
                    weight = (self.DCP_prior - torch.min(self.DCP_prior)) / (
                        torch.max(self.DCP_prior) - torch.min(self.DCP_prior)
                    ).clamp_min(1e-6)
                    weight_resize = F.interpolate(
                        self.DCP_prior, (new_w, new_h)
                    ).reshape(-1, 1)
                    weight = modified_sigmoid_adjusted(
                        weight, a=config.guidance.others.loss_DCP_weight
                    )
                no_sky_index = (weight_resize != 0.1)[:, 0]
                if torch.sum(index) == 0:
                    return feat.sum() * 0.0
                if config.guidance.others.feat_proj:
                    if config.guidance.others.OT_sky:
                        (s, c) = feat_proj[cond_example_ids[0]][:, index].shape
                        X2 = (
                            feat_proj[cond_appearance_ids[0]][:, index]
                            .float()
                            .reshape(-1, c)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        X1 = (
                            feat_proj[cond_control_ids[0]][:, index]
                            .float()
                            .reshape(-1, c)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        (s, c) = feat_proj[cond_example_ids[0]][no_sky_index][
                            :, index
                        ].shape
                        X2 = (
                            feat_proj[cond_appearance_ids[0]][no_sky_index][:, index]
                            .float()
                            .reshape(-1, c)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        X1 = (
                            feat_proj[cond_control_ids[0]][no_sky_index][:, index]
                            .float()
                            .reshape(-1, c)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    (s, c) = feat[cond_example_ids[0]].shape
                    X2 = (
                        feat[cond_appearance_ids[0]]
                        .float()
                        .reshape(-1, c)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    X1 = (
                        feat[cond_control_ids[0]]
                        .float()
                        .reshape(-1, c)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if len(X1) < 2 or len(X2) < 2:
                    return feat.sum() * 0.0
                if config.guidance.others.OT:
                    nb = min(len(X1), 500)
                    rng = np.random.RandomState(42)
                    idx1 = rng.randint(X1.shape[0], size=(nb,))
                    idx2 = rng.randint(X2.shape[0], size=(nb,))
                    Xs = X1[idx1, :]
                    Xt = X2[idx2, :]
                    ot_emd = ot.da.EMDTransport()
                    ot_emd.fit(Xs=Xs, Xt=Xt)
                    transp_Xs_emd = ot_emd.transform(Xs=X1)
                    Image_emd = transp_Xs_emd.reshape(-1, c)
                self.alpha = config.guidance.others.OT_momentum
                if self.alpha == 0:
                    self.alpha = (1 - torch.sum(weight_resize) / (new_h * new_w)).clamp(
                        0.5, 0.8
                    )
                if config.guidance.others.feat_proj:
                    if config.guidance.others.OT:
                        if config.guidance.others.OT_sky:
                            label = self.alpha * feat_proj[cond_control_ids[0]][
                                :, index
                            ] + (1 - self.alpha) * torch.tensor(Image_emd).to(
                                self._execution_device
                            )
                        else:
                            label = self.alpha * feat_proj[cond_control_ids[0]][
                                no_sky_index
                            ][:, index] + (1 - self.alpha) * torch.tensor(Image_emd).to(
                                self._execution_device
                            )
                    elif config.guidance.others.OT_sky:
                        label = feat_proj[cond_appearance_ids[0]][:, index]
                    else:
                        label = feat_proj[cond_appearance_ids[0]][no_sky_index][
                            :, index
                        ]
                elif config.guidance.others.OT:
                    label = self.alpha * feat[cond_control_ids[0]] + (
                        1 - self.alpha
                    ) * torch.tensor(Image_emd).to(self._execution_device)
                else:
                    label = feat[cond_appearance_ids[0]]
            if config.guidance.others.feat_proj:
                if config.guidance.others.OT_sky:
                    proj_control = feat_proj[cond_control_ids[0], :, index]
                else:
                    proj_control = feat_proj[cond_control_ids[0], :, index][
                        no_sky_index
                    ]
            else:
                proj_control = feat[cond_control_ids[0]]
            proj_control.shape[0] // 4
            if config.guidance.others.OT_sky and config.guidance.others.OT:
                label_4D = label.reshape(1, new_w, new_h, -1).permute([0, 3, 1, 2])
                proj_control_4D = proj_control.reshape(1, new_w, new_h, -1).permute(
                    [0, 3, 1, 2]
                )
            (mean_example, cov_example) = compute_mean_cov(
                feat[cond_example_ids]
                .reshape(1, new_w, new_h, -1)
                .permute([0, 3, 1, 2])
            )
            (mean_app, cov_app) = compute_mean_cov(
                feat[cond_appearance_ids]
                .reshape(1, new_w, new_h, -1)
                .permute([0, 3, 1, 2])
            )
            if (
                config.guidance.others.cov_sharp != 0
                and config.guidance.others.OT_sky
                and config.guidance.others.OT
            ):
                proj_control_4D = torch.bmm(
                    proj_control_4D.reshape(1, -1, new_w * new_h).float(),
                    torch.softmax(
                        config.guidance.others.cov_sharp * cov_example.float(), dim=1
                    ),
                ).reshape(1, -1, new_w, new_h)
                label_4D = torch.bmm(
                    label_4D.reshape(1, -1, new_w * new_h).float(),
                    torch.softmax(
                        config.guidance.others.cov_sharp * cov_example.float(), dim=1
                    ),
                ).reshape(1, -1, new_w, new_h)
            if config.guidance.others.OT_sky and config.guidance.others.OT:
                if config.guidance.others.loss_weight_size_scale:
                    proj_control = F.interpolate(
                        proj_control_4D,
                        (self.DCP_prior.shape[2], self.DCP_prior.shape[3]),
                        mode="bilinear",
                    )
                    label = F.interpolate(
                        label_4D,
                        (self.DCP_prior.shape[2], self.DCP_prior.shape[3]),
                        mode="bilinear",
                    )
                else:
                    weight = F.interpolate(weight, (new_w, new_h), mode="bilinear")
                    label = label_4D
                    proj_control = proj_control_4D
            elif config.guidance.others.OT_sky:
                weight = F.interpolate(weight, (new_w, new_h), mode="bilinear").reshape(
                    -1, 1
                )
            else:
                weight = F.interpolate(weight, (new_w, new_h), mode="bilinear").reshape(
                    -1, 1
                )[no_sky_index]
            temp_loss = config.guidance.others.loss_dehaze_weight * (
                F.l1_loss if config.guidance.others.qa_loss == "l1" else F.mse_loss
            )(weight * label, weight * proj_control)
            loss.append(temp_loss)
        loss = torch.stack(loss).sum()
        return loss

    def compute_attn_pca_loss(
        self, cond_control_ids, cond_example_ids, cond_appearance_ids, step_i, config
    ):
        combined_list = cond_example_ids + cond_control_ids + cond_appearance_ids
        new_cond_example_ids = np.arange(len(cond_example_ids)).tolist()
        new_cond_control_ids = np.arange(
            len(cond_example_ids), len(cond_control_ids) + len(cond_example_ids)
        ).tolist()
        new_cond_appearance_ids = np.arange(
            len(cond_control_ids) + len(cond_example_ids), len(combined_list)
        ).tolist()
        pca_loss = []
        step_pca_info: dict = self.loaded_pca_info[step_i]
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if (
                module_name == "Attention"
                and "attn1" in name
                and ("attentions" in name)
                and _classify_blocks(self.guidance_config.pca_guidance.blocks, name)
            ):
                try:
                    select_feature = self.guidance_config.pca_guidance.select_feature
                except (AttributeError, KeyError):
                    select_feature = "key"
                self.current_step = step_i
                if select_feature == "key":
                    self.save_name = name
                    key: torch.Tensor = module.processor.key[combined_list]
                    key_pca_info: dict = step_pca_info["attn_key"][name]
                    module_pca_loss = self._compute_feat_loss(
                        key,
                        key_pca_info,
                        new_cond_control_ids,
                        new_cond_example_ids,
                        new_cond_appearance_ids,
                        step_i,
                        reg_included=True,
                        reg_feature=[key],
                    )
                elif select_feature == "query":
                    query: torch.Tensor = module.processor.query[combined_list]
                    query_pca_info: dict = step_pca_info["attn_query"][name]
                    module_pca_loss = self._compute_feat_loss(
                        query,
                        query_pca_info,
                        new_cond_control_ids,
                        new_cond_example_ids,
                        new_cond_appearance_ids,
                        step_i,
                        reg_included=True,
                        reg_feature=[query],
                    )
                else:
                    value: torch.Tensor = module.processor.value[combined_list]
                    value_pca_info: dict = step_pca_info.get(
                        "attn_value", step_pca_info.get("attn_key", {})
                    )[name]
                    module_pca_loss = self._compute_feat_loss(
                        value,
                        value_pca_info,
                        new_cond_control_ids,
                        new_cond_example_ids,
                        new_cond_appearance_ids,
                        step_i,
                        config,
                        reg_included=True,
                        reg_feature=[value],
                    )
                pca_loss.append(module_pca_loss)
        weight = float(self.guidance_config.pca_guidance.weight)
        if (
            self.guidance_config.pca_guidance.warm_up.apply
            and step_i < self.guidance_config.pca_guidance.warm_up.end_step
        ):
            weight = weight * (
                step_i / self.guidance_config.pca_guidance.warm_up.end_step
            )
        pca_loss = torch.stack(pca_loss).mean() * weight
        return pca_loss

    @torch.no_grad()
    def invert(
        self,
        img: Union[List[PIL.Image.Image], PIL.Image.Image] = None,
        inversion_config: omegaconf.dictconfig = None,
    ):
        # Inversion uses the original Diffusers modules, as in a fresh run.py
        # pipeline. Restore them when reusing a pipeline after feature injection.
        if not hasattr(self, "_inversion_processors"):
            self._inversion_processors = dict(self.unet.attn_processors)
            self._inversion_resnets = [
                (module, module.forward)
                for block in self.unet.up_blocks
                for module in block.resnets
            ]
        self.unet.set_attn_processor(self._inversion_processors.copy())
        for module, forward in self._inversion_resnets:
            module.forward = forward
        select_inversion_method = inversion_config["method"]
        assert select_inversion_method in ["DDIM", "NTI", "NPI"], (
            "Inversion method not supported, please select from ['DDIM', 'NTI', 'NPI']"
        )
        if select_inversion_method == "DDIM":
            self.inverse_scheduler = DDIMInverseScheduler.from_config(
                self.scheduler.config
            )
            if inversion_config.fixed_size is not None:
                img_size = inversion_config.fixed_size
                if isinstance(img, PIL.Image.Image):
                    img = img.resize(img_size)
            if isinstance(img, PIL.Image.Image):
                print("Image size: ", img.size)
                self.img_size: tuple = img.size
            else:
                raise NotImplementedError(
                    "Inversion with a list of images not supported yet"
                )
            prompt: str = inversion_config.prompt
            import time

            start_time = time.time()
            (inv_latents, _, all_latent, prompt_embeds) = self.ddim_inversion(
                prompt,
                image=img,
                num_inference_steps=inversion_config.num_inference_steps,
                return_dict=False,
            )
            end_time = time.time()
            print("Inversion time", end_time - start_time)
            img_data: Dict = {
                "prompt": prompt,
                "all_latents": all_latent,
                "img": PILtoTensor(img),
                "pil_img": img,
                "prompt_embeds": prompt_embeds,
            }
            return img_data
        else:
            raise NotImplementedError("Inversion method not implemented yet")
        pass

    def prepare_image_latents(
        self, image, batch_size, dtype, device, generator=None, mean=False
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        image = image.to(device=device)
        if image.shape[1] == 4:
            latents = image
        else:
            with torch.autocast(
                device_type=self._execution_device.type, dtype=torch.float32
            ):
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
                if isinstance(generator, list):
                    latents = [
                        self.vae.encode(image[i : i + 1]).latent_dist.sample(
                            generator[i]
                        )
                        for i in range(batch_size)
                    ]
                    latents = torch.cat(latents, dim=0)
                elif mean:
                    latents = self.vae.encode(image).latent_dist.mean
                else:
                    latents = self.vae.encode(image).latent_dist.sample(generator)
                latents = self.vae.config.scaling_factor * latents
        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                deprecation_message = f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning."
                deprecate(
                    "len(prompt) != len(image)",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)
        return latents

    @torch.no_grad()
    def ddim_inversion(
        self,
        prompt: Optional[str] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        image = self.image_processor.preprocess(image)
        latents = self.prepare_image_latents(
            image, batch_size, self.vae.dtype, device, generator
        )
        prompt_embeds = self.l_text_encoder(
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=248,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
        )[0]
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps
        all_latents = {}
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            with torch.autocast(
                device_type=self._execution_device.type, dtype=torch.float32
            ):
                for i, t in enumerate(timesteps):
                    timestep_key = t.detach().cpu().item()
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.inverse_scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=prompt_embeds
                    ).sample
                    if do_classifier_free_guidance:
                        (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    latents = self.inverse_scheduler.step(
                        noise_pred, t, latents
                    ).prev_sample
                    all_latents[timestep_key] = latents.detach().cpu()
                    if i == len(timesteps) - 1 or (
                        i + 1 > num_warmup_steps
                        and (i + 1) % self.inverse_scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
        inverted_latents = latents.detach().clone()
        image = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return (inverted_latents, image, all_latents, prompt_embeds)
        return None
