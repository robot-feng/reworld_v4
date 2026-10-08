"""Optional RGB/feature-map visualization for the offline evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def _image(example: dict[str, Any], time_index: int, view_index: int) -> np.ndarray:
    views = example["trajectory"]["images"][time_index]
    image = views[view_index]
    return np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)


def _feature_view(values: Tensor, sample_index: int, view_index: int) -> Tensor:
    if values.ndim == 4:
        if view_index != 0:
            raise IndexError("single-view features only support view_index=0")
        return values[sample_index]
    if values.ndim == 5:
        return values[sample_index, view_index]
    raise ValueError(f"expected [B,C,H,W] or [B,V,C,H,W], got {tuple(values.shape)}")


def _rms(values: Tensor) -> np.ndarray:
    return values.float().square().mean(dim=0).sqrt().cpu().numpy()


def _resize(values: np.ndarray, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.asarray(
        Image.fromarray(values.astype(np.float32)).resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
    )


def save_visualization(
    example: dict[str, Any],
    output: dict[str, Tensor],
    sample_index: int,
    output_path: Path,
    *,
    view_index: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    current_image = _image(example, 0, view_index)
    future_image = _image(example, -1, view_index)
    predicted = _feature_view(output["predicted_feature_delta"], sample_index, view_index)
    target = _feature_view(output["target_feature_delta"], sample_index, view_index)
    predicted_map = _resize(_rms(predicted), current_image)
    target_map = _resize(_rms(target), current_image)
    error_map = _resize(_rms(predicted - target), current_image)
    cosine_map = torch.nn.functional.cosine_similarity(
        predicted.float(), target.float(), dim=0, eps=1e-8
    ).cpu().numpy()

    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(current_image)
    axes[0, 0].set_title("Current RGB")
    axes[0, 1].imshow(future_image)
    axes[0, 1].set_title("Target future RGB")
    axes[0, 2].imshow(current_image)
    axes[0, 2].imshow(predicted_map, cmap="magma", alpha=0.58)
    axes[0, 2].set_title("Predicted residual RMS")
    axes[1, 0].imshow(current_image)
    axes[1, 0].imshow(target_map, cmap="magma", alpha=0.58)
    axes[1, 0].set_title("Target residual RMS")
    axes[1, 1].imshow(error_map, cmap="magma")
    axes[1, 1].set_title("Residual error RMS")
    axes[1, 2].imshow(cosine_map, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 2].set_title("Per-patch residual cosine")
    for axis in axes.flat:
        axis.axis("off")
    indices = example["trajectory"]["observation_indices"]
    figure.suptitle(f"view={view_index} | horizons={indices}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
