"""Public single-image API with reusable frozen diffusion and Long-CLIP models."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import os
import time
import warnings
import numpy as np
import torch
from PIL import Image, ImageOps
from _engine import FreeControlSDPipeline
from attention import clear_features
from basis import DEFAULT_BASIS, load_basis
from config import InferenceConfig
from correction import correct
from longclip_model import build_model
from scheduler import CustomDDIMScheduler

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_MODEL = str(CHECKPOINTS / "sd15")


@dataclass
class FreeDehazeOutput:
    image: Image.Image
    dehazed: Image.Image
    perception: Image.Image
    reconstruction: Image.Image
    correction_mask: Image.Image
    metadata: dict
    input: Image.Image


class FreeDehazePipeline:
    """Single-image inference. CUDA is required.

    Gradients are enabled for latent guidance and the optional per-image corrector;
    SD and Long-CLIP parameters are frozen. Do not wrap calls in inference_mode().
    """

    def __init__(self, engine, config, model_id, basis_manifest, basis_path):
        self.engine = engine
        self.config = config
        self.model_id = str(model_id)
        self.basis_manifest = basis_manifest
        self.basis_path = str(basis_path)

    @classmethod
    def from_pretrained(
        cls,
        model_id=DEFAULT_MODEL,
        *,
        longclip_path=CHECKPOINTS / "longclip-L.pt",
        pca_path=DEFAULT_BASIS,
        config=None,
        device="cuda",
        local_files_only=True,
        unet_path=None,
    ):
        config = (config or InferenceConfig()).validate()
        if config.deterministic:
            # Must be configured before the first CUDA BLAS operation.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "FreeDehaze requires a CUDA-enabled PyTorch installation and NVIDIA GPU"
            )
        if not Path(longclip_path).is_file():
            raise FileNotFoundError(
                f"Long-CLIP checkpoint not found: {longclip_path}. Place longclip-L.pt in checkpoints/."
            )
        scheduler = CustomDDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", local_files_only=local_files_only
        )
        scheduler.set_timesteps(config.steps)
        if (
            scheduler.config.steps_offset != 1
            or scheduler.config.timestep_spacing != "leading"
        ):
            raise ValueError(
                "Expected SD 1.5 scheduler: leading timestep spacing and steps_offset=1"
            )
        basis, manifest = load_basis(
            pca_path, scheduler.timesteps.tolist(), config.components
        )
        extra = {}
        if unet_path is not None:
            from diffusers import UNet2DConditionModel

            if not Path(unet_path).is_dir():
                raise FileNotFoundError(
                    f"Requested UNet checkpoint does not exist: {unet_path}"
                )
            extra["unet"] = UNet2DConditionModel.from_pretrained(
                unet_path, torch_dtype=torch.float16, local_files_only=local_files_only
            )
        if config.preset == "ohaze" and unet_path is None:
            warnings.warn(
                "OHAZE preset uses base SD 1.5; paper smoke experiments used a separate DreamBooth UNet."
            )
        engine = FreeControlSDPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
            **extra,
        ).to(device)
        if engine.unet.config.cross_attention_dim != 768:
            raise ValueError(
                "This release supports Stable Diffusion 1.5 (768 text channels) only"
            )
        engine.scheduler = scheduler
        state = torch.load(longclip_path, map_location="cpu", weights_only=True)
        if state["ln_final.weight"].shape[0] != 768:
            raise ValueError("Use Long-CLIP-L, not Long-CLIP-B")
        longclip = build_model(state).to(device).eval().requires_grad_(False)
        for module in (engine.unet, engine.vae, engine.text_encoder):
            module.eval().requires_grad_(False)
        engine.l_text_encoder = longclip.encode_text
        engine.loaded_pca_info = basis
        # Basis data is validated once, then reused across images.
        engine.load_pca_info = lambda: None
        return cls(engine, config, model_id, manifest, pca_path)

    def __call__(self, image, prompt="", *, seed=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Remove torch.inference_mode(): FreeDehaze needs latent gradients"
            )
        cfg = self.config
        chosen_seed = cfg.seed if seed is None else seed
        if type(chosen_seed) is not int or not 0 <= chosen_seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        elif isinstance(image, Image.Image):
            image = ImageOps.exif_transpose(image).convert("RGB")
        else:
            raise TypeError("image must be a PIL image or a filesystem path")
        original_size = image.size
        original_image = image.copy()
        image = image.resize((cfg.size, cfg.size), Image.Resampling.BICUBIC)
        random.seed(chosen_seed)
        np.random.seed(chosen_seed)
        torch.manual_seed(chosen_seed)
        torch.cuda.manual_seed_all(chosen_seed)
        old_benchmark = torch.backends.cudnn.benchmark
        old_cudnn = torch.backends.cudnn.deterministic
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.cuda.synchronize(self.engine._execution_device)
        start = time.perf_counter()
        config = cfg.engine_config()
        try:
            if cfg.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
            inverted = self.engine.invert(image, config.data.inversion)
            with torch.enable_grad():
                images = self.engine(
                    prompt=prompt,
                    lq_img=image,
                    negative_prompt=cfg.negative_prompt,
                    num_inference_steps=cfg.steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=torch.Generator().manual_seed(chosen_seed),
                    config=config,
                    inverted_data={"condition_input": [inverted]},
                ).images
            for output in images:
                if output.size != image.size:
                    raise RuntimeError("Unexpected diffusion output dimensions")
            clear_features(self.engine.unet)
            # The reconstruction is the reconstruction branch's decoded output,
            # matching the *_exam image consumed by the original correction scripts.
            final, mask = correct(
                image,
                images[2],
                images[0],
                steps=cfg.correction_steps,
                lr=cfg.correction_lr,
                local=cfg.correction_local,
                device=self.engine._execution_device,
            )
        finally:
            clear_features(self.engine.unet)
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_cudnn
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.cuda.synchronize(self.engine._execution_device)
        if final.size != original_size:
            final = final.resize(original_size, Image.Resampling.LANCZOS)
        images = [
            output.resize(original_size, Image.Resampling.LANCZOS)
            if output.size != original_size
            else output
            for output in images
        ]
        if mask.size != original_size:
            mask = mask.resize(original_size, Image.Resampling.BILINEAR)
        metadata = {
            "config": asdict(cfg) | {"seed": chosen_seed},
            "prompt": prompt,
            "model": self.model_id,
            "basis_source_sha256": self.basis_manifest["source_sha256"],
            "original_size": original_size,
            "output_size": final.size,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        return FreeDehazeOutput(
            final, images[0], images[1], images[2], mask, metadata, original_image
        )
