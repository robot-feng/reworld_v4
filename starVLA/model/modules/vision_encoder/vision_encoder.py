"""Slim unified vision encoder for DINO/DINOv2/DINOv3, TIPSv2 and RADIO.

Core contract:
- get_vision_encoder(...) -> nn.Module
- encoder.prepare_input(images) supports nested PIL images [B][V] or tensors
- encoder(x) -> patch tokens [B*V, N, C]
- encoder(x, return_dict=True) -> tokens + feature_map + grid

Model priority:
1. explicit model_path
2. local model_root candidates
3. official HF/timm id, unless local_files_only=True
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torchvision import transforms


# starVLA/model/modules/vision_encoder/vision_encoder.py -> starVLA root
_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else Path.cwd()
DEFAULT_MODEL_ROOT = Path(os.getenv("ARVLA_MODEL_ROOT", REPO_ROOT / "playground" / "Pretrained_models"))

DEFAULT_MODEL_IDS = {
    "dino": "timm/vit_base_patch16_dinov3.lvd1689m",
    "dinov2": "timm/vit_small_patch14_dinov2.lvd142m",
    "dinov3": "timm/vit_base_patch16_dinov3.lvd1689m",
    "tips": "google/tipsv2-b14",
    "tipsv2": "google/tipsv2-b14",
    "radio": "nvidia/C-RADIOv4-SO400M",
}

ALIASES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3-vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3-vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_base": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3-base": "vit_base_patch16_dinov3.lvd1689m",
}

LOCAL_NAMES = {
    "dino": ("vit_base_patch16_dinov3.lvd1689m", "dinov3-base", "dinov3_vitb16", "vit_small_patch16_dinov3.lvd1689m", "dinov3-vits16", "dinov3_vits16"),
    "dinov2": ("vit_small_patch14_dinov2.lvd142m", "dinov2_vits14"),
    "dinov3": ("vit_base_patch16_dinov3.lvd1689m", "dinov3-base", "dinov3_vitb16", "vit_small_patch16_dinov3.lvd1689m", "dinov3-vits16", "dinov3_vits16"),
    "tips": ("tipsv2-b14", "google--tipsv2-b14", "models--google--tipsv2-b14"),
    "tipsv2": ("tipsv2-b14", "google--tipsv2-b14", "models--google--tipsv2-b14"),
    "radio": ("C-RADIOv4-SO400M", "nvidia--C-RADIOv4-SO400M", "models--nvidia--C-RADIOv4-SO400M"),
}

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt")
WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "open_clip_pytorch_model.bin",
    "checkpoint.safetensors",
    "checkpoint.pth",
    "model.pth",
    "model.pt",
)


@dataclass
class VisionEncoderOutput:
    tokens: torch.Tensor       # [B, N, C]
    feature_map: torch.Tensor  # [B, C, Gh, Gw]
    grid: tuple[int, int]
    model_id: str
    backend: str


def _cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _pair(x: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else (int(x[0]), int(x[1]))


def _abs(path: str | os.PathLike) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _device(module: nn.Module) -> torch.device:
    return next(module.parameters(), torch.empty(0)).device


def _dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters(), torch.empty(0, dtype=torch.float32)).dtype


def _encoder_type(x: Optional[str]) -> str:
    x = str(x or "dino").lower()
    if "radio" in x:
        return "radio"
    if "tips" in x:
        return "tips"
    if "dinov2" in x:
        return "dinov2"
    if "dinov3" in x:
        return "dinov3"
    if "dino" in x or x.startswith("vit_"):
        return "dino"
    return x


def _model_name(model_name: Optional[str], encoder_type: str) -> str:
    return ALIASES.get(str(model_name), str(model_name)) if model_name else DEFAULT_MODEL_IDS[encoder_type]


def _local_candidates(model_name: str, encoder_type: str) -> list[str]:
    names: list[str] = []

    def add(name: str) -> None:
        if name and name not in names:
            names.append(name)

    name = model_name.removeprefix("hf_hub:").removeprefix("timm/")
    add(model_name)
    add(name)
    if "/" in name:
        owner, repo = name.split("/", 1)
        add(repo)
        add(f"{owner}--{repo}")
        add(f"models--{owner}--{repo}")
    for item in LOCAL_NAMES.get(encoder_type, ()):  # curated short names
        add(item)
    return names


def _find_weight(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix in WEIGHT_SUFFIXES:
        return path
    if not path.is_dir():
        return None
    for name in WEIGHT_NAMES:
        if (path / name).exists():
            return path / name
    for suffix in WEIGHT_SUFFIXES:
        found = sorted(path.glob(f"*{suffix}"))
        if found:
            return found[0]
    return None


def _is_hf_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _resolve_model(
    encoder_type: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    model_root: str | os.PathLike = DEFAULT_MODEL_ROOT,
    local_files_only: bool = False,
) -> tuple[str, Optional[Path], str]:
    """Return (model_ref, local_path, source)."""
    name = _model_name(model_name, encoder_type)

    if model_path:
        path = _abs(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Vision encoder path does not exist: {path}")
        return str(path), path, "explicit"

    root = _abs(model_root)
    for local_name in _local_candidates(name, encoder_type):
        for path in (root / local_name, *(root / f"{local_name}{s}" for s in WEIGHT_SUFFIXES)):
            if path.exists():
                return str(path), path, "local"

    if local_files_only:
        raise FileNotFoundError(
            f"No local {encoder_type} model found under {root}. "
            f"model_name={name!r}. Set model_path/model_root or disable local_files_only."
        )
    return name, None, "remote"


def _hidden(config: Any, default: int) -> int:
    for key in ("hidden_size", "embed_dim", "num_features", "vision_embed_dim"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    vc = getattr(config, "vision_config", None)
    value = vc.get("hidden_size") if isinstance(vc, dict) else getattr(vc, "hidden_size", None)
    return int(value) if value is not None else default


def _patch_tokens(output: Any, patch_n: Optional[int] = None) -> torch.Tensor:
    """Extract [B,N,C] spatial tokens from common timm/HF outputs."""
    keys = ("x_norm_patchtokens", "patch_tokens", "patchtokens", "features", "spatial_features", "last_hidden_state")
    if isinstance(output, dict):
        output = next((output[k] for k in keys if k in output), output)
    else:
        output = next((getattr(output, k) for k in keys if getattr(output, k, None) is not None), output)

    if not torch.is_tensor(output):
        raise RuntimeError(f"Cannot extract patch tokens from {type(output)}")

    if output.ndim == 4:  # [B,C,H,W] or [B,H,W,C]
        if patch_n and output.shape[1] * output.shape[2] == patch_n:  # NHWC
            output = output.reshape(output.shape[0], patch_n, output.shape[3])
        else:  # NCHW
            output = output.flatten(2).transpose(1, 2).contiguous()
    if output.ndim != 3:
        raise RuntimeError(f"Expected tokens [B,N,C], got {tuple(output.shape)}")
    return output[:, -patch_n:] if patch_n and output.shape[1] >= patch_n else output


def _grid(tokens: torch.Tensor, image_hw: Optional[tuple[int, int]], patch_size: int) -> tuple[int, int]:
    if image_hw:
        gh, gw = image_hw[0] // patch_size, image_hw[1] // patch_size
        if gh * gw == tokens.shape[1]:
            return gh, gw
    side = int(tokens.shape[1] ** 0.5)
    if side * side == tokens.shape[1]:
        return side, side
    raise ValueError(f"Cannot infer grid from {tokens.shape[1]} tokens; pass image_hw.")


def _timm_name(name: str) -> str:
    name = ALIASES.get(name, name)
    if name.startswith("hf_hub:"):
        return name
    if name.startswith("vit_") and "dinov3" in name and "." in name:
        return f"hf_hub:timm/{name}"
    return f"hf_hub:{name}" if name.startswith("timm/") else name


def _dino_arch(name: str) -> str:
    return ALIASES.get(name, name).removeprefix("hf_hub:timm/").removeprefix("timm/")


def _from_pretrained(cls: Any, model_id: str, trust_remote_code: bool, local_files_only: bool) -> Any:
    try:
        return cls.from_pretrained(model_id, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    except TypeError:  # Some processors do not accept all kwargs.
        return cls.from_pretrained(model_id, trust_remote_code=trust_remote_code)


class VisionEncoderBase(nn.Module):
    patch_size: int = 16
    num_channels: int = 0
    model_id: str = ""
    backend: str = ""

    def __init__(self, image_size: int | tuple[int, int] = 224, freeze: bool = False):
        super().__init__()
        self.image_size = _pair(image_size)
        self.freeze = freeze

    def _freeze(self) -> None:
        if self.freeze:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    def prepare_input(self, images: Any) -> torch.Tensor:
        if torch.is_tensor(images):
            x = images.flatten(0, 1) if images.ndim == 5 else images
            if x.ndim != 4:
                raise ValueError(f"Expected [B,V,C,H,W] or [B,C,H,W], got {tuple(images.shape)}")
            return x.to(device=_device(self), dtype=_dtype(self) if x.is_floating_point() else None)

        # Accept both [B][V] and flat [V].
        if images and not isinstance(images[0], (list, tuple)):
            images = [images]
        xs = [[self.transform(img.convert("RGB") if hasattr(img, "convert") else img) for img in views] for views in images]
        x = torch.stack([torch.stack(v) for v in xs]).flatten(0, 1)
        return x.to(device=_device(self), dtype=_dtype(self))

    prepare_dino_input = prepare_input

    def tokens_to_map(self, tokens: torch.Tensor, image_hw: Optional[tuple[int, int]] = None) -> torch.Tensor:
        gh, gw = _grid(tokens, image_hw, self.patch_size)
        return tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()

    def _out(self, tokens: torch.Tensor, pixels: torch.Tensor, return_dict: bool) -> torch.Tensor | VisionEncoderOutput:
        if not return_dict:
            return tokens
        image_hw = (int(pixels.shape[-2]), int(pixels.shape[-1]))
        return VisionEncoderOutput(tokens, self.tokens_to_map(tokens, image_hw), _grid(tokens, image_hw, self.patch_size), self.model_id, self.backend)


class DINOVisionEncoder(VisionEncoderBase):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        model_root: str | os.PathLike = DEFAULT_MODEL_ROOT,
        image_size: Optional[int | tuple[int, int]] = None,
        encoder_type: str = "dino",
        pretrained: bool = True,
        freeze: bool = False,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__(image_size or 224, freeze)
        encoder_type = _encoder_type(model_name if encoder_type == "dino" and model_name else encoder_type)
        requested = _model_name(model_name, encoder_type)
        ref, path, source = _resolve_model(encoder_type, requested, model_path, model_root, local_files_only)

        self.model_id, self.model_source = ref, source
        self.checkpoint_path: Optional[str] = None
        self.backend = "hf" if path and _is_hf_dir(path) else "timm" if (path or ref.startswith(("timm/", "hf_hub:timm/", "vit_"))) else "hf"

        if path and self.backend == "timm":
            weight = _find_weight(path)
            if path.is_file():
                weight = path
            if not weight:
                raise FileNotFoundError(f"No DINO weight/config found in {path}")
            self.checkpoint_path = str(weight)
            self.model_id = _dino_arch(requested)

        if self.backend == "timm":
            import timm
            from timm.data import create_transform, resolve_model_data_config

            self.body = timm.create_model(
                _timm_name(self.model_id),
                pretrained=bool(pretrained and self.checkpoint_path is None),
                num_classes=0,
                checkpoint_path=self.checkpoint_path or "",
            ).eval()
            ps = getattr(getattr(self.body, "patch_embed", None), "patch_size", 16)
            self.patch_size = int(ps[0] if isinstance(ps, tuple) else ps)
            self.num_channels = int(getattr(self.body, "num_features", 0) or getattr(self.body, "embed_dim", 0))
            cfg = resolve_model_data_config(self.body)
            if image_size is not None:
                cfg["input_size"] = (3, *self.image_size)
            self.transform = create_transform(**cfg, is_training=False)
        else:
            from transformers import AutoModel

            self.body = _from_pretrained(AutoModel, self.model_id, trust_remote_code, local_files_only).eval()
            config = getattr(self.body, "config", None)
            self.patch_size = int(getattr(config, "patch_size", 16))
            self.num_channels = _hidden(config, 384)
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self._freeze()

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = False) -> torch.Tensor | VisionEncoderOutput:
        patch_n = (pixel_values.shape[-2] // self.patch_size) * (pixel_values.shape[-1] // self.patch_size)
        raw = self.body.forward_features(pixel_values) if self.backend == "timm" else self.body(pixel_values=pixel_values)
        return self._out(_patch_tokens(raw, patch_n), pixel_values, return_dict)


class TIPSVisionEncoder(VisionEncoderBase):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        model_root: str | os.PathLike = DEFAULT_MODEL_ROOT,
        image_size: int | tuple[int, int] = 448,
        freeze: bool = False,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__(image_size, freeze)
        from transformers import AutoModel

        self.model_id, _, self.model_source = _resolve_model("tips", model_name, model_path, model_root, local_files_only)
        self.backend = "hf"
        self.body = _from_pretrained(AutoModel, self.model_id, trust_remote_code, local_files_only).eval()
        vc = getattr(self.body.config, "vision_config", None)
        self.patch_size = int(_cfg(vc, "patch_size", 14))
        self.num_channels = int(_cfg(vc, "hidden_size", _hidden(self.body.config, 768)))
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
        self._freeze()

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = False) -> torch.Tensor | VisionEncoderOutput:
        return self._out(_patch_tokens(self.body.encode_image(pixel_values)), pixel_values, return_dict)

    def encode_text(self, texts: list[str]) -> Any:
        return self.body.encode_text(texts)


class RADIOVisionEncoder(VisionEncoderBase):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        model_root: str | os.PathLike = DEFAULT_MODEL_ROOT,
        image_size: Optional[int | tuple[int, int]] = None,
        freeze: bool = False,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__(image_size or (256, 256), freeze)
        from transformers import AutoModel, CLIPImageProcessor

        self.model_id, _, self.model_source = _resolve_model("radio", model_name, model_path, model_root, local_files_only)
        self.backend = "hf"
        self.processor = _from_pretrained(CLIPImageProcessor, self.model_id, False, local_files_only)
        self.body = _from_pretrained(AutoModel, self.model_id, trust_remote_code, local_files_only).eval()
        self.patch_size = int(getattr(self.body, "patch_size", getattr(self.body.config, "patch_size", 16)))
        self.num_channels = int(getattr(getattr(self.body, "radio_model", None), "embed_dim", 1152))
        self.transform = None
        self._freeze()

    def prepare_input(self, images: Any) -> torch.Tensor:
        if torch.is_tensor(images):
            return super().prepare_input(images)
        if images and not isinstance(images[0], (list, tuple)):
            images = [images]
        flat = [img.convert("RGB") if hasattr(img, "convert") else img for views in images for img in views]
        x = self.processor(images=flat, return_tensors="pt", do_resize=True).pixel_values
        return x.to(device=_device(self), dtype=_dtype(self))

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = False) -> torch.Tensor | VisionEncoderOutput:
        out = self.body(pixel_values)
        raw = out[1] if isinstance(out, tuple) else next(
            (getattr(out, k) for k in ("features", "spatial_features", "last_hidden_state") if getattr(out, k, None) is not None),
            None,
        )
        if raw is None and isinstance(out, dict):
            raw = next((out[k] for k in ("features", "spatial_features", "last_hidden_state") if k in out), None)
        if raw is None:
            raise RuntimeError(f"Cannot extract RADIO features from {type(out)}")
        patch_n = (pixel_values.shape[-2] // self.patch_size) * (pixel_values.shape[-1] // self.patch_size)
        return self._out(_patch_tokens(raw, patch_n), pixel_values, return_dict)


def get_vision_encoder(
    config: Any = None,
    encoder_type: Optional[str] = None,
    backone_name: Optional[str] = None,   # keep original typo compatibility
    backbone_name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    backbone = backbone_name or backone_name or _cfg(config, "backbone_name") or _cfg(config, "backone_name") or _cfg(config, "dino_backbone")
    model_name = model_name or _cfg(config, "model_name") or _cfg(config, "model_id") or backbone
    encoder_type = _encoder_type(encoder_type or _cfg(config, "encoder_type") or model_name)

    common = dict(
        model_name=model_name,
        model_path=model_path or _cfg(config, "model_path") or _cfg(config, "local_path"),
        model_root=_cfg(config, "model_root", kwargs.pop("model_root", DEFAULT_MODEL_ROOT)),
        freeze=bool(_cfg(config, "freeze", kwargs.pop("freeze", False))),
        trust_remote_code=bool(_cfg(config, "trust_remote_code", kwargs.pop("trust_remote_code", True))),
        local_files_only=bool(_cfg(config, "local_files_only", kwargs.pop("local_files_only", False))),
    )
    image_size = _cfg(config, "image_size", kwargs.pop("image_size", None))

    if encoder_type in {"dino", "dinov2", "dinov3"}:
        return DINOVisionEncoder(
            **common,
            image_size=image_size,
            encoder_type=encoder_type,
            pretrained=bool(_cfg(config, "pretrained", kwargs.pop("pretrained", True))),
            **kwargs,
        )
    if encoder_type in {"tips", "tipsv2"}:
        return TIPSVisionEncoder(**common, image_size=image_size or 448, **kwargs)
    if encoder_type == "radio":
        return RADIOVisionEncoder(**common, image_size=image_size, **kwargs)
    raise ValueError(f"Unsupported vision encoder type: {encoder_type}")


DINOv3VisionEncoder = DINOVisionEncoder
DINOv2VisionEncoder = DINOVisionEncoder


if __name__ == "__main__":
    from PIL import Image

    encoder = get_vision_encoder(
        encoder_type="dino",
        model_name="vit_base_patch16_dinov3.lvd1689m",
        image_size=256,
        pretrained=True,
        freeze=True,
    ).eval()

    images = [[Image.new("RGB", (256, 256), "white"), Image.new("RGB", (256, 256), "black")]]
    x = encoder.prepare_input(images)
    with torch.inference_mode():
        out = encoder(x, return_dict=True)

    print(f"model: {encoder.model_id}")
    print(f"source: {getattr(encoder, 'model_source', None)}")
    print(f"backend: {encoder.backend}")
    print(f"checkpoint: {getattr(encoder, 'checkpoint_path', None)}")
    print(f"input: {tuple(x.shape)}")
    print(f"tokens: {tuple(out.tokens.shape)}")
    print(f"feature_map: {tuple(out.feature_map.shape)}, grid={out.grid}")
    print(f"num_channels: {encoder.num_channels}, patch_size={encoder.patch_size}")
