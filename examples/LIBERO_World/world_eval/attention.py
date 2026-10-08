"""Read-only attention probes for the semantic-prefill residual world model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AttentionSnapshot:
    residual_to_current: Tensor
    residual_to_residual: Tensor
    action_to_residual: Tensor


class WorldAttentionRecorder:
    """Recover aggregate attention maps with hooks, leaving model code untouched."""

    def __init__(self, model) -> None:
        self.model = model
        self.handles = []
        self.clear()

    @staticmethod
    def _probabilities(attention, hidden_states: Tensor, context: Tensor) -> Tensor:
        batch, queries = hidden_states.shape[:2]
        keys = context.shape[1]
        heads = attention.heads
        query = attention.to_q(hidden_states).reshape(batch, queries, heads, -1).transpose(1, 2)
        key = attention.to_k(context).reshape(batch, keys, heads, -1).transpose(1, 2)
        if getattr(attention, "norm_q", None) is not None:
            query = attention.norm_q(query)
        if getattr(attention, "norm_k", None) is not None:
            key = attention.norm_k(key)
        return torch.softmax(torch.matmul(query.float(), key.float().transpose(-1, -2)) * attention.scale, dim=-1)

    def _record_residual(self, _module, args, kwargs) -> None:
        probabilities = self._probabilities(_module, args[0], kwargs["encoder_hidden_states"])
        token_count = args[0].shape[1]
        aggregate = probabilities.mean(dim=(1, 2))[0].detach().cpu()
        self.residual_to_current.append(aggregate[:token_count])
        self.residual_to_residual.append(aggregate[token_count:])

    def _record_action(self, _module, args, kwargs) -> None:
        probabilities = self._probabilities(_module, args[0], kwargs["encoder_hidden_states"])
        self.action_to_residual.append(probabilities.mean(dim=(1, 2))[0].detach().cpu())

    def __enter__(self):
        for block in self.model.blocks:
            self.handles.append(block.attention.register_forward_pre_hook(self._record_residual, with_kwargs=True))
        for block in self.model.action_resamplers:
            self.handles.append(block.cross_attention.register_forward_pre_hook(self._record_action, with_kwargs=True))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.residual_to_current: list[Tensor] = []
        self.residual_to_residual: list[Tensor] = []
        self.action_to_residual: list[Tensor] = []

    def snapshot(self) -> AttentionSnapshot:
        expected = len(self.model.blocks)
        if not all(len(values) == expected for values in (
            self.residual_to_current, self.residual_to_residual, self.action_to_residual
        )):
            raise RuntimeError("attention hooks did not observe every residual-world layer")
        return AttentionSnapshot(
            torch.stack(self.residual_to_current),
            torch.stack(self.residual_to_residual),
            torch.stack(self.action_to_residual),
        )


def spatialize(values: np.ndarray, spatial_shape: tuple[int, int], view_count: int | None) -> np.ndarray:
    views = view_count or 1
    return values.reshape(*values.shape[:-1], views, *spatial_shape)


def _cosine_to_first(attention: np.ndarray) -> np.ndarray:
    flattened = attention.reshape(attention.shape[0], attention.shape[1], -1)
    reference = flattened[:1]
    numerator = (flattened * reference).sum(-1)
    denominator = np.linalg.norm(flattened, axis=-1) * np.linalg.norm(reference, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def save_attention_outputs(
    output_dir: Path,
    horizons: np.ndarray,
    current_image: np.ndarray,
    residual_to_current: np.ndarray,
    residual_to_residual: np.ndarray,
    action_to_residual: np.ndarray,
    task_change_maps: np.ndarray,
    action_horizon: int,
    view_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "attention_sweep.npz",
        horizons=horizons,
        current_image=current_image,
        residual_to_current=residual_to_current,
        residual_to_residual=residual_to_residual,
        action_to_residual=action_to_residual,
        task_change_maps=task_change_maps,
    )

    layers = np.arange(1, action_to_residual.shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for layer in range(action_to_residual.shape[1]):
        axes[0, 0].plot(horizons, _cosine_to_first(action_to_residual)[:, layer], label=f"layer {layer + 1}")
        axes[0, 1].plot(horizons, residual_to_current[:, layer].sum(axis=(-3, -2, -1)), label=f"layer {layer + 1}")
    probability = action_to_residual / np.maximum(action_to_residual.sum(axis=(-3, -2, -1), keepdims=True), 1e-12)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=(-3, -2, -1))
    axes[1, 0].plot(horizons, entropy)
    axes[1, 1].plot(horizons, np.sqrt(np.mean(np.square(task_change_maps), axis=(-3, -2, -1))))
    axes[0, 0].set(title="Action-attention cosine similarity to h=1", xlabel="horizon", ylabel="cosine")
    axes[0, 1].set(title="Residual attention mass on current features", xlabel="horizon", ylabel="mass")
    axes[1, 0].set(title="Action-attention entropy", xlabel="horizon", ylabel="entropy")
    axes[1, 1].set(title="Predicted task-change RMS", xlabel="horizon", ylabel="RMS")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2)
    axes[0, 1].legend(ncol=2)
    axes[1, 0].legend([f"layer {layer}" for layer in layers], ncol=2)
    fig.savefig(output_dir / "attention_summary.png", dpi=180)
    plt.close(fig)

    candidates = [1, 2, 4, action_horizon, 16, 32, 64, 128, 256, int(horizons[-1])]
    selected = sorted({value for value in candidates if value in set(horizons.tolist())})
    indices = [int(np.searchsorted(horizons, value)) for value in selected]
    grid = plt.figure(figsize=(2.5 + 2.2 * len(selected), 7), constrained_layout=True)
    spec = grid.add_gridspec(3, len(selected) + 1, width_ratios=[1.25] + [1] * len(selected))
    current_axis = grid.add_subplot(spec[:, 0])
    current_axis.imshow(current_image)
    current_axis.set_title("Current RGB\n(first frame)")
    current_axis.axis("off")
    rows = (
        (task_change_maps, "Task-change RMS", "magma"),
        (action_to_residual[:, -1], "Action → residual\n(last layer)", "viridis"),
        (residual_to_current[:, -1], "Residual → current\n(last layer)", "viridis"),
    )
    for row, (values, label, cmap) in enumerate(rows):
        scale = values[indices, view_index]
        lower, upper = float(scale.min()), float(np.percentile(scale, 99.5))
        for column, (horizon, index) in enumerate(zip(selected, indices), start=1):
            axis = grid.add_subplot(spec[row, column])
            axis.imshow(values[index, view_index], cmap=cmap, vmin=lower, vmax=upper)
            axis.set_title(f"h={horizon}" if row == 0 else "")
            if column == 1:
                axis.set_ylabel(label, labelpad=8)
            axis.axis("off")
    grid.savefig(output_dir / "attention_contact_sheet.png", dpi=180)
    plt.close(grid)
