"""Offline action/world gradient-conflict diagnosis for QwenResidualWorldPrefill."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from examples.LIBERO_World.world_eval.runner import (
    _seed,
    load_checkpoint,
    resolve_checkpoint,
    resolve_config,
    run_directory,
)


def _freeze_configured_modules(model: torch.nn.Module, freeze_modules: str) -> list[str]:
    frozen: list[str] = []
    for path in (part.strip() for part in freeze_modules.split(",")):
        if not path:
            continue
        module: Any = model
        for name in path.split("."):
            module = getattr(module, name)
        module.requires_grad_(False)
        frozen.append(path)
    return frozen


def _suite_name(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    for suite in ("spatial", "object", "goal", "10"):
        if f"libero_{suite}" in lowered:
            return f"libero_{suite}"
    return dataset_name or "unknown"


def _gradient_metrics(
    action_loss: torch.Tensor,
    world_loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, float | int | bool]:
    parameters = [parameter for _, parameter in named_parameters]
    action_grads = torch.autograd.grad(
        action_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    world_grads = torch.autograd.grad(
        world_loss,
        parameters,
        allow_unused=True,
    )

    device = action_loss.device
    zero = torch.zeros((), device=device, dtype=torch.float64)
    shared_dot = zero.clone()
    shared_action_sq = zero.clone()
    shared_world_sq = zero.clone()
    global_action_sq = zero.clone()
    global_world_sq = zero.clone()
    global_combined_sq = zero.clone()
    shared_parameter_tensors = 0
    shared_parameter_elements = 0

    for (_, parameter), action_grad, world_grad in zip(named_parameters, action_grads, world_grads):
        action = None if action_grad is None else action_grad.detach().float()
        world = None if world_grad is None else world_grad.detach().float()
        if action is not None:
            action_flat = action.reshape(-1)
            global_action_sq += torch.dot(action_flat, action_flat).double()
        if world is not None:
            world_flat = world.reshape(-1)
            global_world_sq += torch.dot(world_flat, world_flat).double()

        if action is None:
            combined = world
        elif world is None:
            combined = action
        else:
            combined = action + world
            action_flat = action.reshape(-1)
            world_flat = world.reshape(-1)
            shared_dot += torch.dot(action_flat, world_flat).double()
            shared_action_sq += torch.dot(action_flat, action_flat).double()
            shared_world_sq += torch.dot(world_flat, world_flat).double()
            shared_parameter_tensors += 1
            shared_parameter_elements += parameter.numel()
        if combined is not None:
            combined_flat = combined.reshape(-1)
            global_combined_sq += torch.dot(combined_flat, combined_flat).double()

    shared_action_norm = math.sqrt(float(shared_action_sq.item()))
    shared_world_norm = math.sqrt(float(shared_world_sq.item()))
    denominator = shared_action_norm * shared_world_norm
    cosine = float(shared_dot.item()) / denominator if denominator > 0 else math.nan
    return {
        "grad_cosine": cosine,
        "shared_action_grad_norm": shared_action_norm,
        "shared_weighted_world_grad_norm": shared_world_norm,
        "weighted_world_action_grad_norm_ratio": (
            shared_world_norm / shared_action_norm if shared_action_norm > 0 else math.inf
        ),
        "global_action_grad_norm": math.sqrt(float(global_action_sq.item())),
        "global_weighted_world_grad_norm": math.sqrt(float(global_world_sq.item())),
        "global_combined_grad_norm_before_clip": math.sqrt(float(global_combined_sq.item())),
        "shared_parameter_tensors": shared_parameter_tensors,
        "shared_parameter_elements": shared_parameter_elements,
    }


def _summary(records: Iterable[dict[str, Any]], clip_threshold: float) -> dict[str, Any]:
    records = list(records)
    fields = (
        "action_loss",
        "world_loss",
        "weighted_world_loss",
        "grad_cosine",
        "shared_action_grad_norm",
        "shared_weighted_world_grad_norm",
        "weighted_world_action_grad_norm_ratio",
        "global_combined_grad_norm_before_clip",
        "seconds",
    )
    result: dict[str, Any] = {"count": len(records)}
    for field in fields:
        values = [float(record[field]) for record in records if math.isfinite(float(record[field]))]
        if values:
            result[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    finite_cosines = [float(record["grad_cosine"]) for record in records if math.isfinite(record["grad_cosine"])]
    result["negative_cosine_fraction"] = (
        sum(value < 0 for value in finite_cosines) / len(finite_cosines) if finite_cosines else math.nan
    )
    result["clip_trigger_fraction"] = (
        sum(record["global_combined_grad_norm_before_clip"] > clip_threshold for record in records) / len(records)
        if records
        else math.nan
    )
    return result


def main(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from starVLA.model.framework.base_framework import build_framework

    checkpoint = resolve_checkpoint(args.checkpoint)
    config_path = resolve_config(checkpoint, args.config_yaml)
    config = OmegaConf.load(config_path)
    if args.data_mix:
        config.datasets.vla_data.data_mix = args.data_mix
    world_weight = (
        float(config.framework.world_loss_weight)
        if args.world_weight is None
        else float(args.world_weight)
    )
    if world_weight <= 0:
        raise ValueError("world weight must be positive for a weighted gradient comparison")

    _seed(args.seed)
    model = build_framework(config)
    load_checkpoint(model, checkpoint)
    freeze_spec = str(config.trainer.get("freeze_modules", ""))
    frozen = _freeze_configured_modules(model, freeze_spec)
    if not hasattr(model, "residual_world") or not hasattr(model, "action_model"):
        raise TypeError("gradient diagnosis requires a residual-world framework with an action head")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).train()
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise RuntimeError("no trainable parameters remain after applying freeze_modules")

    dataset = get_vla_dataset(config.datasets.vla_data, mode="val", seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    clip_threshold = float(config.trainer.get("gradient_clipping", config.trainer.get("max_grad_norm", 1.0)))
    output_dir = args.output_dir or (
        run_directory(checkpoint)
        / "evaluation"
        / checkpoint.stem.removesuffix("_pytorch_model")
        / "gradient_conflict"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = 0

    for scan_index, examples in enumerate(loader):
        if scan_index >= args.max_scanned_samples:
            break
        dataset_name = str(examples[0].get("trajectory", {}).get("dataset_name", ""))
        suite = _suite_name(dataset_name)
        if suite_counts.get(suite, 0) >= args.samples_per_suite:
            if len(suite_counts) >= args.expected_suites and all(
                count >= args.samples_per_suite for count in suite_counts.values()
            ):
                break
            continue

        batch_started = time.perf_counter()
        _seed(args.seed + scan_index)
        autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(examples)
            action_loss = output["action_loss"]
            raw_world_loss = output["world_loss"]
            weighted_world_loss = world_weight * raw_world_loss
        metrics = _gradient_metrics(action_loss, weighted_world_loss, named_parameters)
        trajectory = examples[0]["trajectory"]
        offsets = trajectory["observation_indices"]
        record = {
            "scan_index": scan_index,
            "suite": suite,
            "dataset_name": dataset_name,
            "horizon_mid": int(offsets[1]),
            "horizon_future": int(offsets[-1]),
            "action_loss": float(action_loss.detach().float().cpu()),
            "world_loss": float(raw_world_loss.detach().float().cpu()),
            "weighted_world_loss": float(weighted_world_loss.detach().float().cpu()),
            "seconds": time.perf_counter() - batch_started,
            **metrics,
        }
        records.append(record)
        suite_counts[suite] = suite_counts.get(suite, 0) + 1
        print(
            f"[{len(records):03d}] {suite:<15} "
            f"cos={record['grad_cosine']:+.4f} "
            f"ratio={record['weighted_world_action_grad_norm_ratio']:.3f} "
            f"combined={record['global_combined_grad_norm_before_clip']:.3f}",
            flush=True,
        )
        del output, action_loss, raw_world_loss, weighted_world_loss

        if len(suite_counts) >= args.expected_suites and all(
            count >= args.samples_per_suite for count in suite_counts.values()
        ):
            break

    grouped = {
        suite: _summary((record for record in records if record["suite"] == suite), clip_threshold)
        for suite in sorted({record["suite"] for record in records})
    }
    result = {
        "metadata": {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "data_mix": str(config.datasets.vla_data.data_mix),
            "world_weight": world_weight,
            "clip_threshold": clip_threshold,
            "samples_per_suite": args.samples_per_suite,
            "seed": args.seed,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
            "frozen_modules": frozen,
            "elapsed_seconds": time.perf_counter() - started,
            "baseline_cuda_allocated_bytes": baseline_allocated,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        },
        "all": _summary(records, clip_threshold),
        "by_suite": grouped,
    }
    (output_dir / "gradient_conflict.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved gradient diagnosis to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-mix", default="libero_residual_world_all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--samples-per-suite", type=int, default=20)
    parser.add_argument("--expected-suites", type=int, default=4)
    parser.add_argument("--max-scanned-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--world-weight", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
