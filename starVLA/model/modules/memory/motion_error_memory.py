"""Differentiable, delayed-feedback linear fast memory (no autograd inner loop)."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MotionErrorMemory(nn.Module):
    def __init__(self, feature_dim: int, semantic_dim: int, dim: int = 128,
                 grid_size: int = 7, inner_lr: float = 0.01,
                 forget_factor: float = 0.995, gate_init: float = 1e-3,
                 value_mode: str = "motion_error"):
        super().__init__()
        if min(feature_dim, semantic_dim, dim, grid_size) < 1:
            raise ValueError("memory dimensions must be positive")
        if not 0 < inner_lr <= 1 or not 0 <= forget_factor <= 1:
            raise ValueError("require 0 < inner_lr <= 1 and 0 <= forget_factor <= 1")
        if value_mode not in {"motion", "error", "motion_error"}:
            raise ValueError("value_mode must be motion|error|motion_error")
        self.dim, self.grid_size = dim, grid_size
        self.inner_lr, self.forget_factor = inner_lr, forget_factor
        self.value_mode = value_mode
        # Three coordinates (view, y, x) and log horizon; semantic projection
        # is performed before broadcasting over spatial tokens.
        self.semantic = nn.Linear(semantic_dim, dim)
        self.key = nn.Linear(2 * feature_dim + dim + 4, dim)
        self.query = nn.Linear(2 * feature_dim + dim + 4, dim)
        self.value = nn.Linear(feature_dim * (2 if value_mode == "motion_error" else 1), dim)
        self.decoder = nn.Linear(dim, feature_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @staticmethod
    def _linear(layer: nn.Linear, x: Tensor) -> Tensor:
        # Explicit casts preserve FP32 arithmetic even after model.to(bfloat16).
        return F.linear(x.float(), layer.weight.float(),
                        None if layer.bias is None else layer.bias.float())

    def pool(self, features: Tensor) -> Tensor:
        if features.ndim == 4:
            features = features[:, None]
        if features.ndim != 5:
            raise ValueError("memory features must be [B,C,H,W] or [B,V,C,H,W]")
        b, v, c, _, _ = features.shape
        pooled = F.adaptive_avg_pool2d(features.float().flatten(0, 1), self.grid_size)
        return pooled.reshape(b, v, c, -1).permute(0, 1, 3, 2).reshape(b, -1, c)

    def address(self, semantic: Tensor, current: Tensor, delta: Tensor,
                horizons: Tensor, mask: Tensor | None = None, *, query: bool = False) -> Tensor:
        with torch.autocast(current.device.type, enabled=False):
            z, r = self.pool(current), self.pool(delta)
            s = semantic.float()
            if mask is None:
                s = s.mean(1)
            else:
                weights = mask.float().unsqueeze(-1)
                s = (s * weights).sum(1) / weights.sum(1).clamp_min(1)
            s = self._linear(self.semantic, s)[:, None].expand(-1, z.shape[1], -1)
            views = current.shape[1] if current.ndim == 5 else 1
            grid = torch.linspace(-1, 1, self.grid_size, device=z.device)
            view, y, x = torch.meshgrid(torch.arange(views, device=z.device).float(), grid, grid, indexing="ij")
            position = torch.stack((view, y, x), -1).reshape(1, -1, 3).expand(z.shape[0], -1, -1)
            h = horizons.float().log1p()[:, None, None].expand(-1, z.shape[1], 1)
            inputs = torch.cat((F.layer_norm(z, (z.shape[-1],)),
                                F.layer_norm(r, (r.shape[-1],)), s, position, h), -1)
            return F.normalize(self._linear(self.query if query else self.key, inputs), dim=-1, eps=1e-6)

    def values(self, motion: Tensor, error: Tensor) -> Tensor:
        with torch.autocast(motion.device.type, enabled=False):
            inputs = {"motion": (motion,), "error": (error,), "motion_error": (motion, error)}[self.value_mode]
            return self._linear(self.value, torch.cat(inputs, -1))

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.dim, self.dim, device=device, dtype=torch.float32)

    def write(self, keys: Tensor, values: Tensor, weights: Tensor, valid: Tensor) -> Tensor:
        """One averaged update across *all* feedback tokens due at this observation.

        Rows without feedback are unchanged, including their forgetting factor.
        """
        with torch.autocast(weights.device.type, enabled=False):
            keys, values, weights = keys.float(), values.float(), weights.float()
            error = values - keys @ weights.transpose(-1, -2)
            scale = valid.float() / keys.square().sum(-1).clamp_min(1e-6)
            update = (error * scale[..., None]).transpose(-1, -2) @ keys
            count = valid.sum(-1)
            update = update / count.clamp_min(1)[:, None, None]
            updated = self.forget_factor * weights + self.inner_lr * update
            return torch.where((count > 0)[:, None, None], updated, weights)

    def read(self, queries: Tensor, weights: Tensor, reference: Tensor) -> Tensor:
        with torch.autocast(reference.device.type, enabled=False):
            tokens = self._linear(self.decoder, queries.float() @ weights.float().transpose(-1, -2))
            b = reference.shape[0]
            v = reference.shape[1] if reference.ndim == 5 else 1
            g = self.grid_size
            maps = tokens.reshape(b * v, g, g, -1).permute(0, 3, 1, 2)
            maps = F.interpolate(maps, size=reference.shape[-2:], mode="bilinear", align_corners=False)
            correction = maps.reshape(b, v, *maps.shape[1:]) * self.gate.float().tanh()
            # Keep the small update in FP32 through residual addition and loss.
            # Casting here makes early updates disappear when added to BF16 R0.
            return correction[:, 0] if reference.ndim == 4 else correction
