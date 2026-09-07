"""Portable, validated inference settings; no dataset paths or UI dependencies."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class InferenceConfig:
    preset: str = "rtts"
    steps: int = 100
    size: int = 512
    seed: int = 2038
    deterministic: bool = True
    guidance_scale: float = 7.5
    guidance_start: float = 0.1
    guidance_end: float = 0.5
    guidance_weight: float = 50.0
    components: int = 64
    weight_inject: float = 0.0
    qk_inject: float = 0.99
    conv_inject: float = 0.6
    perception_stop: int = 400
    dehaze_cfg_stop: int = 100
    dcp_kernel: int = 15
    dcp_weight: float = 5.0
    dehaze_weight: float = 10.0
    sky: bool = True
    optimal_transport: bool = False
    include_sky_in_qa: bool = False
    source_momentum: float = 0.0
    covariance_scale: float = 5.0
    qa_loss: str = "mse"
    blur_guidance: bool = False
    correction_steps: int = 100
    correction_lr: float = 0.001
    correction_local: bool = True
    negative_prompt: str = (
        "foggy, fog, haze, hazy, mist, smog, smoke, smoggy, cartoonish, blur, "
        "blurry, anime, artifacts, unrealistic, unclear edges, non-photorealistic, "
        "unnatural colors, anime-style, inconsistent textures, stylized, unnatural "
        "transitions, comic, unnatural outlines, anime-influence, motion blur, "
        "anime-aesthetic, cartoon-aestheti"
    )

    @classmethod
    def from_preset(cls, name="rtts", **overrides):
        presets = {
            "rtts": {},
            "fla": dict(
                dcp_weight=50.0,
                dehaze_weight=50.0,
                weight_inject=20.0,
                perception_stop=0,
            ),
            "ohaze": dict(
                dcp_weight=0.1,
                dehaze_weight=100.0,
                sky=False,
                blur_guidance=True,
                optimal_transport=True,
                include_sky_in_qa=True,
                perception_stop=0,
                dehaze_cfg_stop=0,
            ),
            # Mechanism-oriented setting, not a claim of reproducing the paper tables.
            "paper": dict(
                optimal_transport=True,
                include_sky_in_qa=True,
                qa_loss="l1",
                source_momentum=0.2,
                weight_inject=1.0,
                qk_inject=0.95,
                dehaze_cfg_stop=-1,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset {name!r}; choose {', '.join(presets)}")
        return cls(preset=name, **(presets[name] | overrides)).validate()

    @classmethod
    def from_yaml(cls, path, **overrides):
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return cls.from_preset(values.pop("preset", "rtts"), **(values | overrides))

    def validate(self):
        for name in (
            "deterministic",
            "sky",
            "optimal_transport",
            "include_sky_in_qa",
            "blur_guidance",
            "correction_local",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        for name in (
            "steps",
            "size",
            "seed",
            "components",
            "dcp_kernel",
            "correction_steps",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 10 <= self.steps <= 1000 or 1000 % self.steps:
            raise ValueError(
                "steps must divide 1000 exactly and be between 10 and 1000"
            )
        if self.size < 64 or self.size % 64:
            raise ValueError("size must be a positive multiple of 64 (paper: 512)")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not 1 <= self.components <= 1280:
            raise ValueError("components must be in [1, 1280]")
        if self.guidance_scale <= 1:
            raise ValueError("guidance_scale must exceed 1 for the five-branch engine")
        if not 0 <= self.guidance_start < self.guidance_end <= 1:
            raise ValueError("Require 0 <= guidance_start < guidance_end <= 1")
        if int(self.steps * self.guidance_start) == int(self.steps * self.guidance_end):
            raise ValueError("The QA guidance interval contains no sampling steps")
        if self.dcp_kernel < 1 or self.dcp_kernel % 2 != 1:
            raise ValueError("dcp_kernel must be a positive odd integer")
        if self.correction_steps < 0 or self.correction_lr <= 0:
            raise ValueError("correction_steps must be >= 0 and correction_lr > 0")
        for name in ("qk_inject", "conv_inject", "source_momentum"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "guidance_weight",
            "weight_inject",
            "dcp_weight",
            "dehaze_weight",
            "covariance_scale",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.qa_loss not in {"l1", "mse"}:
            raise ValueError("qa_loss must be l1 or mse")
        return self

    def engine_config(self):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "sd_config": {
                    "same_latent": True,
                    "appearnace_same_latent": True,
                    "grad_guidance_scale": 1.0,
                    "pca_paths": [],
                    "steps": self.steps,
                },
                "data": {
                    "inversion": {
                        "method": "DDIM",
                        "fixed_size": None,
                        "prompt": "",
                        "num_inference_steps": self.steps,
                    }
                },
                "guidance": {
                    "pca_guidance": {
                        "start_step": int(self.steps * self.guidance_start),
                        "end_step": int(self.steps * self.guidance_end),
                        "weight": self.guidance_weight,
                        "select_feature": "value",
                        "blocks": ["up_blocks.1"],
                        "structure_guidance": {
                            "n_components": self.components,
                            "normalized": False,
                            "mask_tr": 0.5,
                        },
                        "warm_up": {"apply": False, "end_step": 0},
                    },
                    "cross_attn": {
                        "qk_inject": self.qk_inject,
                        "conv_inject": self.conv_inject,
                        "weight_inject": self.weight_inject,
                        "inject_app_stop_t": self.perception_stop,
                        "res_dict": (
                            {0: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
                            if self.preset == "ohaze"
                            else {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
                        ),
                    },
                    "others": {
                        "DCP_kernal": self.dcp_kernel,
                        "sky": self.sky,
                        "longclip": True,
                        "blur_guidance": self.blur_guidance,
                        "exam_fusion": 500,
                        "dehaze_CFG": self.dehaze_cfg_stop,
                        "cross_prior": False,
                        "feat_proj": True,
                        "OT": self.optimal_transport,
                        "OT_sky": self.include_sky_in_qa,
                        "OT_momentum": self.source_momentum,
                        "loss_DCP_weight": self.dcp_weight,
                        "loss_dehaze_weight": self.dehaze_weight,
                        "cov_sharp": self.covariance_scale,
                        "loss_weight_size_scale": True,
                        "qa_loss": self.qa_loss,
                        "grad_clip": True,
                    },
                },
            }
        )
