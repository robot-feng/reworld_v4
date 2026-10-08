"""Checkpoint and dataloader runner for the isolated world-eval plugin."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import batch_world_metrics, summarize_records
from .visualization import save_visualization


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = list((path / "checkpoints").glob("*.pt")) + list(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {path}")

    def step(candidate: Path) -> int:
        match = re.search(r"steps_(\d+)", candidate.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=lambda candidate: (step(candidate), candidate.stat().st_mtime_ns))


def run_directory(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def resolve_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    root = run_directory(checkpoint)
    for name in ("config.full.yaml", "config.launch.yaml", "config.yaml"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no saved config found under {root}")


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(payload)}")
    for prefix in ("module.", "_forward_module."):
        if payload and all(key.startswith(prefix) for key in payload):
            payload = {key.removeprefix(prefix): value for key, value in payload.items()}
    model.load_state_dict(payload, strict=True)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _record(
    example: dict[str, Any],
    metrics: dict[str, torch.Tensor],
    batch_index: int,
    sample_index: int,
    batch_size: int,
) -> dict[str, Any]:
    trajectory = example["trajectory"]
    offsets = trajectory["observation_indices"]
    frames = trajectory.get("frame_indices", offsets)
    terminal = trajectory.get("terminal_index")
    record = {
        "sample_index": batch_index * batch_size + sample_index,
        "dataset_name": str(trajectory.get("dataset_name", "")),
        "horizon_mid": int(offsets[1]),
        "horizon": int(offsets[-1]),
        "target_is_terminal": terminal is not None and int(frames[-1]) == int(terminal),
    }
    record.update(
        {
            name: float(values[sample_index].detach().cpu())
            for name, values in metrics.items()
        }
    )
    return record


def main(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from starVLA.model.framework.base_framework import build_framework

    started_at = time.perf_counter()
    checkpoint = resolve_checkpoint(args.checkpoint)
    config_path = resolve_config(checkpoint, args.config_yaml)
    config = OmegaConf.load(config_path)
    if args.data_mix:
        config.datasets.vla_data.data_mix = args.data_mix

    _seed(args.seed)
    model = build_framework(config)
    random_core = None
    if args.random_core:
        random_core = {
            name: value.detach().clone()
            for name, value in model.core.state_dict().items()
        }
    load_checkpoint(model, checkpoint)
    if random_core is not None:
        model.core.load_state_dict(random_core, strict=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device=device, dtype=dtype).eval()
    _seed(args.seed)

    dataset = get_vla_dataset(
        data_cfg=config.datasets.vla_data,
        mode="val",
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    label = "random_core" if args.random_core else checkpoint.stem
    output_dir = args.output_dir or run_directory(checkpoint) / "world_evaluation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    visualization_count = 0

    with torch.inference_mode():
        for batch_index, examples in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            output = model.evaluate_world(examples)
            metrics = batch_world_metrics(output)
            for sample_index, example in enumerate(examples):
                record = _record(
                    example,
                    metrics,
                    batch_index,
                    sample_index,
                    args.batch_size,
                )
                record["sample_index"] = len(records)
                records.append(record)
                if visualization_count < args.max_visualizations:
                    save_visualization(
                        example,
                        output,
                        sample_index,
                        output_dir / "visualizations" / f"sample_{visualization_count:04d}.png",
                        view_index=args.view_index,
                    )
                    visualization_count += 1

    runtime = {"elapsed_seconds": time.perf_counter() - started_at}
    if device.type == "cuda":
        runtime.update(
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    result = {
        "metadata": {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "evaluation_mode": "direct_and_self_forced",
            "random_core": args.random_core,
            "seed": args.seed,
            "num_samples": len(records),
            "data_mix": str(config.datasets.vla_data.data_mix),
            "runtime": runtime,
        },
        "groups": summarize_records(records),
    }
    (output_dir / "world_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-mix", default="libero_residual_world_all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--max-visualizations", type=int, default=8)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-core", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
