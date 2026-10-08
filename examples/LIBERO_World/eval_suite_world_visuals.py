#!/usr/bin/env python3
"""Compare semantic-prefill residual worlds on identical LIBERO tasks.

This is a read-only, offline diagnostic.  It deliberately lives next to (but
does not alter) the normal LIBERO rollout evaluator.  One deterministic expert
sample is selected for every unique instruction in each suite, then every
checkpoint is evaluated on that exact 4 x 10 task set.

For each selected camera view the script writes a compact panel containing
RGB, predicted/target residual maps and PCA, residual error, plus the compact
action context actually passed to the action head.  The ``Action resampler``
overlay is the final resampler's attention from compact action queries to the
spatial residual tokens: it is the spatial provenance of that context, rather
than an indirect post-hoc saliency map.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUITES = ("libero_object", "libero_goal", "libero_spatial", "libero_10")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = list((path / "checkpoints").glob("*.pt")) + list(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {path}")
    return max(candidates, key=lambda value: int(re.search(r"steps_(\d+)", value.name).group(1)))


def _run_dir(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _config_path(checkpoint: Path) -> Path:
    for name in ("config.full.yaml", "config.launch.yaml", "config.yaml"):
        candidate = _run_dir(checkpoint) / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config YAML found beside {checkpoint}")


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)}")
    for prefix in ("module.", "_forward_module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load unexpectedly reported incompatible keys")


def _suite(example: dict[str, Any]) -> str:
    trajectory = example.get("trajectory", {})
    name = str(trajectory.get("dataset_name", example.get("temporal_dataset_name", "")))
    return next((suite for suite in SUITES if suite in name), "unknown")


def _task_key(example: dict[str, Any]) -> str:
    # LIBERO's natural-language task is stable across all demonstrations and
    # is a better task key than a random temporal frame index.
    return " ".join(str(example.get("lang", "")).lower().split())


def _image(example: dict[str, Any], time_index: int, view_index: int) -> np.ndarray:
    trajectory = example["trajectory"]
    value = trajectory["images"][time_index][view_index]
    if hasattr(value, "convert"):
        value = value.convert("RGB")
    return np.asarray(value)


def _views(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features[:, None]
    if features.ndim == 5:
        return features
    raise ValueError(f"Expected [B,C,H,W] or [B,V,C,H,W], got {tuple(features.shape)}")


def _pca_rgb(*maps: torch.Tensor) -> list[np.ndarray]:
    """Shared PCA basis, so colours are comparable within one panel only."""
    vectors = [item.detach().float().cpu().permute(1, 2, 0).reshape(-1, item.shape[0]) for item in maps]
    matrix = torch.cat(vectors)
    centered = matrix - matrix.mean(0, keepdim=True)
    count = min(3, centered.shape[0], centered.shape[1])
    _, _, basis = torch.pca_lowrank(centered, q=count, center=False, niter=4)
    projected = centered @ basis
    if count < 3:
        projected = F.pad(projected, (0, 3 - count))
    low, high = torch.quantile(projected, 0.01, 0), torch.quantile(projected, 0.99, 0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    offset, result = 0, []
    for item, vector in zip(maps, vectors, strict=True):
        height, width = item.shape[-2:]
        result.append(projected[offset : offset + len(vector)].reshape(height, width, 3).numpy())
        offset += len(vector)
    return result


def _resize(values: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(values.astype(np.float32)).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR))


def _unit(values: np.ndarray) -> np.ndarray:
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / max(high - low, 1e-8)


def _map_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    # Attention probes are intentionally detached on CPU, while world maps
    # remain on the model device.  Metrics are tiny, so make that boundary
    # explicit instead of moving a probe back onto a busy training GPU.
    first = predicted.detach().float().cpu().flatten()
    second = target.detach().float().cpu().flatten()
    corr = torch.corrcoef(torch.stack((first, second)))[0, 1].nan_to_num().item()
    count = max(1, int(round(first.numel() * 0.1)))
    first_top = torch.topk(first, count).indices
    second_top = torch.topk(second, count).indices
    intersection = torch.isin(first_top, second_top).sum().item()
    return float(corr), float(intersection / (2 * count - intersection))


def _action_context_pca(context: torch.Tensor) -> np.ndarray:
    """PCA of the compact query tokens; its 4xN grid has no image geometry."""
    token_count = context.shape[0]
    rgb = _pca_rgb(context.transpose(0, 1).reshape(context.shape[1], 1, token_count))[0][0]
    columns = min(8, token_count)
    rows = int(np.ceil(token_count / columns))
    padded = np.zeros((rows * columns, 3), dtype=np.float32)
    padded[:token_count] = rgb
    return padded.reshape(rows, columns, 3)


def _save_panel(
    *,
    example: dict[str, Any],
    output: dict[str, torch.Tensor],
    action_contexts: tuple[torch.Tensor, ...],
    action_attention: torch.Tensor,
    sample: int,
    view: int,
    output_path: Path,
    metrics: dict[str, float],
) -> None:
    current_rgb, target_rgb = _image(example, 0, view), _image(example, -1, view)
    current = _views(output["current_vision_features"])[sample, view]
    predicted_future = _views(output["predicted_future_vision_features"])[sample, view]
    target_future = _views(output["target_vision_features"])[sample, view]
    predicted_delta = _views(output["predicted_feature_delta"])[sample, view]
    target_delta = _views(output["target_feature_delta"])[sample, view]
    current_pca, predicted_future_pca, target_future_pca = _pca_rgb(current, predicted_future, target_future)
    predicted_delta_pca, target_delta_pca = _pca_rgb(predicted_delta, target_delta)
    pred_map = predicted_delta.float().square().mean(0).sqrt().cpu().numpy()
    target_map = target_delta.float().square().mean(0).sqrt().cpu().numpy()
    error_map = (predicted_delta - target_delta).float().square().mean(0).sqrt().cpu().numpy()
    cosine = F.cosine_similarity(predicted_delta.float(), target_delta.float(), dim=0).cpu().numpy()
    attention_map = action_attention[-1, view].cpu().numpy()
    context = action_contexts[-1][sample].detach().float().cpu()
    context_pca = _action_context_pca(context)
    context_norm = context.norm(dim=-1).numpy()
    scale = max(float(np.quantile(np.concatenate((pred_map.ravel(), target_map.ravel())), 0.99)), 1e-8)

    def overlay(axis, values: np.ndarray, title: str, cmap: str = "magma", vmax: float | None = None) -> None:
        axis.imshow(current_rgb)
        axis.imshow(_resize(values, current_rgb), cmap=cmap, alpha=0.58, vmin=0, vmax=vmax)
        axis.set_title(title)

    figure, axes = plt.subplots(4, 4, figsize=(14, 13), constrained_layout=True)
    axes[0, 0].imshow(current_rgb); axes[0, 0].set_title("Current RGB")
    axes[0, 1].imshow(target_rgb); axes[0, 1].set_title("Target future RGB")
    overlay(axes[0, 2], pred_map, "Predicted residual RMS", vmax=scale)
    overlay(axes[0, 3], target_map, "Target residual RMS", vmax=scale)
    axes[1, 0].imshow(predicted_delta_pca); axes[1, 0].set_title("Predicted residual PCA")
    axes[1, 1].imshow(target_delta_pca); axes[1, 1].set_title("Target residual PCA")
    axes[1, 2].imshow(error_map, cmap="magma"); axes[1, 2].set_title("Residual error RMS")
    axes[1, 3].imshow(cosine, cmap="coolwarm", vmin=-1, vmax=1); axes[1, 3].set_title("Per-token residual cosine")
    axes[2, 0].imshow(current_pca); axes[2, 0].set_title("Current C-RADIO PCA")
    axes[2, 1].imshow(predicted_future_pca); axes[2, 1].set_title("Predicted future PCA")
    axes[2, 2].imshow(target_future_pca); axes[2, 2].set_title("Target future PCA")
    overlay(axes[2, 3], _unit(attention_map), "Action resampler → residual", cmap="viridis", vmax=1)
    axes[3, 0].imshow(context_pca, interpolation="nearest"); axes[3, 0].set_title("Action context PCA\n(query grid is not spatial)")
    axes[3, 1].bar(np.arange(len(context_norm)), context_norm, color="#4c78a8")
    axes[3, 1].set(title="Final action-context token norms", xlabel="query", ylabel="L2")
    axes[3, 2].axis("off")
    axes[3, 2].text(0, 1, "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items()), va="top", family="monospace", fontsize=9)
    axes[3, 3].imshow(_resize(_unit(attention_map) * target_map, current_rgb), cmap="viridis")
    axes[3, 3].set_title("Attention × target residual\n(not a probability map)")
    for axis in axes.flat:
        axis.axis("off")
    axes[3, 1].axis("on")
    trajectory = example["trajectory"]
    figure.suptitle(
        f"{_suite(example)} | h={trajectory['observation_indices'][-1]} | view={view}\n"
        f"{str(example.get('lang', ''))[:130]}", fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _save_contact_sheet(panels: list[Path], output: Path, title: str) -> None:
    images = [np.asarray(Image.open(path).convert("RGB").resize((280, 260))) for path in panels]
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index < len(images):
            axis.imshow(images[index])
            axis.set_title(f"Task {index:02d}")
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _select_examples(config: Any, *, seed: int, tasks_per_suite: int, workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    dataset = get_vla_dataset(data_cfg=config.datasets.vla_data, mode="val", seed=seed)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0)
    selected: dict[str, dict[str, dict[str, Any]]] = {suite: {} for suite in SUITES}
    for examples in loader:
        for example in examples:
            suite, key = _suite(example), _task_key(example)
            if suite in selected and key and len(selected[suite]) < tasks_per_suite:
                selected[suite].setdefault(key, example)
        if all(len(values) >= tasks_per_suite for values in selected.values()):
            result = [example for suite in SUITES for _, example in sorted(selected[suite].items())]
            manifest = []
            for index, example in enumerate(result):
                trajectory = example["trajectory"]
                manifest.append({
                    "selection_index": index, "suite": _suite(example), "lang": str(example["lang"]),
                    "dataset_name": str(trajectory.get("dataset_name", "")),
                    "observation_indices": [int(value) for value in trajectory["observation_indices"]],
                    "frame_indices": [int(value) for value in trajectory.get("frame_indices", [])],
                })
            return result, manifest
    missing = {suite: tasks_per_suite - len(values) for suite, values in selected.items() if len(values) < tasks_per_suite}
    raise RuntimeError(f"Validation set did not expose enough unique tasks: {missing}")


def _direct_world(model: Any, examples: list[dict[str, Any]]):
    """The direct h future path, with the action states preserved for inspection."""
    from examples.LIBERO_World.world_eval.attention import WorldAttentionRecorder

    batch = model._prepare_batch(examples, training=True)
    target = model._encode_vision(batch.future_images)
    semantic, semantic_mask = model._encode_vlm(batch.vlm_images, batch.instructions)
    current = model._encode_vision(batch.world_images)
    with WorldAttentionRecorder(model.residual_world) as recorder:
        prefill = model.residual_world.prefill(semantic, current, semantic_mask)
        raw = model.residual_world.reason(current, batch.horizons, prefill, return_action_states=True)
    output = dict(raw)
    output.update(
        target_vision_features=target,
        target_feature_delta=(target - current).detach(),
    )
    snapshot = recorder.snapshot()
    spatial = tuple(int(value) for value in _views(current).shape[-2:])
    view_count = _views(current).shape[1]
    action_attention = snapshot.action_to_residual.reshape(len(snapshot.action_to_residual), view_count, *spatial)
    contexts, _ = model._action_contexts(raw["action_hidden_states"])
    return output, tuple(raw["action_hidden_states"]), action_attention, contexts


def _sample_metrics(output: dict[str, torch.Tensor], attention: torch.Tensor, contexts: tuple[torch.Tensor, ...]) -> dict[str, float]:
    predicted, target = output["predicted_feature_delta"].float(), output["target_feature_delta"].float()
    delta_mse = (predicted - target).square().mean().item()
    target_energy = target.square().mean().clamp_min(1e-8).item()
    cosine = F.cosine_similarity(predicted.flatten(), target.flatten(), dim=0).item()
    pred_map = predicted.square().mean(-3).sqrt()[0]
    target_map = target.square().mean(-3).sqrt()[0]
    map_corr, map_iou = _map_metrics(pred_map, target_map)
    attention_corr, attention_iou = _map_metrics(attention[-1, 0], target_map[0])
    context = contexts[-1][0].float()
    layer_cosine = F.cosine_similarity(contexts[0][0].float().flatten(), context.flatten(), dim=0).item()
    return {
        "delta_nmse": delta_mse / target_energy,
        "delta_cosine": cosine,
        "task_map_corr": map_corr,
        "task_map_top10_iou": map_iou,
        "action_resampler_target_corr": attention_corr,
        "action_resampler_target_top10_iou": attention_iou,
        "final_context_norm": context.norm(dim=-1).mean().item(),
        "context_first_last_cosine": layer_cosine,
    }


def _write_comparison(output_dir: Path, records: list[dict[str, Any]], labels: list[str]) -> None:
    metrics = [key for key in records[0] if key not in {"model", "suite", "task", "lang", "panel", "npz"}]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["suite"])].append(record)
    rows = []
    for (model, suite), values in grouped.items():
        rows.append({"model": model, "suite": suite, "count": len(values), **{key: float(np.mean([item[key] for item in values])) for key in metrics}})
    with (output_dir / "suite_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "suite_comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    plotted = ("delta_nmse", "delta_cosine", "task_map_corr", "task_map_top10_iou", "action_resampler_target_corr", "action_resampler_target_top10_iou")
    width = 0.23
    for axis, metric in zip(axes.flat, plotted, strict=True):
        positions = np.arange(len(SUITES))
        for index, label in enumerate(labels):
            values = [next(row[metric] for row in rows if row["model"] == label and row["suite"] == suite) for suite in SUITES]
            axis.bar(positions + (index - (len(labels) - 1) / 2) * width, values, width, label=label)
        axis.set(title=metric, xticks=positions, xticklabels=[suite.removeprefix("libero_") for suite in SUITES])
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(output_dir / "suite_comparison.png", dpi=170)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="Repeat exactly three times.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(args: argparse.Namespace) -> None:
    if len(args.checkpoint) != 3:
        raise ValueError("This comparison needs exactly three --checkpoint arguments")
    if args.tasks_per_suite < 1:
        raise ValueError("--tasks-per-suite must be positive")
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework

    checkpoints = [_checkpoint(path) for path in args.checkpoint]
    labels = [path.parent.parent.name for path in checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError("Checkpoint run directory names must be distinct")
    configs = [OmegaConf.load(_config_path(path)) for path in checkpoints]
    first = configs[0]
    expected = (str(first.framework.name), str(first.datasets.vla_data.data_mix), str(first.framework.vision_encoder.model_name))
    for config in configs[1:]:
        observed = (str(config.framework.name), str(config.datasets.vla_data.data_mix), str(config.framework.vision_encoder.model_name))
        if observed != expected:
            raise ValueError(f"Checkpoints do not share the same data/model contract: {expected} != {observed}")

    _seed(args.seed)
    selected, manifest = _select_examples(first, seed=args.seed, tasks_per_suite=args.tasks_per_suite, workers=args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for label, checkpoint, config in zip(labels, checkpoints, configs, strict=True):
        _seed(args.seed)
        model = build_framework(config)
        _load_checkpoint(model, checkpoint)
        model = model.to(device=device, dtype=dtype).eval()
        model_records: list[dict[str, Any]] = []
        model_dir = args.output_dir / "models" / label
        with torch.inference_mode():
            for selection_index, example in enumerate(selected):
                output, hidden_states, attention, contexts = _direct_world(model, [example])
                metrics = _sample_metrics(output, attention, hidden_states)
                suite = _suite(example)
                task = selection_index % args.tasks_per_suite
                task_dir = model_dir / suite / f"task_{task:02d}"
                panel_paths = []
                for view in range(_views(output["current_vision_features"]).shape[1]):
                    panel = task_dir / f"view_{view:02d}.png"
                    _save_panel(example=example, output=output, action_contexts=hidden_states, action_attention=attention,
                                sample=0, view=view, output_path=panel, metrics=metrics)
                    panel_paths.append(panel)
                npz = task_dir / "tensors.npz"
                np.savez_compressed(
                    npz,
                    # This is exactly the list handed to action_model.  With
                    # interleaved self-attention, a world layer can occur more
                    # than once; retaining the repeats makes that wiring
                    # explicit rather than silently collapsing it.
                    action_head_contexts=np.stack([item[0].detach().float().cpu().numpy().astype(np.float16) for item in contexts]),
                    action_context_layers=np.stack([item[0].detach().float().cpu().numpy().astype(np.float16) for item in hidden_states]),
                    action_resampler_attention=attention.numpy().astype(np.float16),
                    predicted_residual=output["predicted_feature_delta"][0].detach().float().cpu().numpy().astype(np.float16),
                    target_residual=output["target_feature_delta"][0].detach().float().cpu().numpy().astype(np.float16),
                )
                record = {"model": label, "suite": suite, "task": task, "lang": str(example["lang"]),
                          "panel": str(panel_paths[0].relative_to(args.output_dir)), "npz": str(npz.relative_to(args.output_dir)), **metrics}
                model_records.append(record)
                print(f"[{label}] {selection_index + 1}/{len(selected)} {suite} task={task}", flush=True)
        for suite in SUITES:
            panels = [model_dir / suite / f"task_{index:02d}" / "view_00.png" for index in range(args.tasks_per_suite)]
            _save_contact_sheet(panels, model_dir / suite / "contact_sheet_view_00.png", f"{label} | {suite} | primary view")
        with (model_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
            for record in model_records:
                stream.write(json.dumps(record) + "\n")
        all_records.extend(model_records)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_comparison(args.output_dir, all_records, labels)
    metadata = {"checkpoints": [str(item) for item in checkpoints], "labels": labels, "seed": args.seed,
                "tasks_per_suite": args.tasks_per_suite, "dtype": str(dtype), "elapsed_seconds": time.perf_counter() - started,
                "interpretation": {"action_resampler_attention": "Final resampler query-to-residual attention; spatial provenance of the compact context passed to the action head.",
                                   "action_context_pca": "PCA of compact action tokens. Its grid is query order, not image position; compare structure/norms within a model, not RGB colours across plots."}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved suite visual comparison to {args.output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
