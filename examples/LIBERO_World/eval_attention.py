"""Sweep residual-world attention over inference horizons for one first-frame observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from examples.LIBERO_World.world_eval.attention import WorldAttentionRecorder, save_attention_outputs, spatialize
from examples.LIBERO_World.world_eval.runner import _seed, load_checkpoint, resolve_checkpoint, resolve_config, run_directory


def main(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from starVLA.model.framework.base_framework import build_framework

    checkpoint = resolve_checkpoint(args.checkpoint)
    config_path = resolve_config(checkpoint, args.config_yaml)
    config = OmegaConf.load(config_path)
    config.datasets.vla_data.data_mix = args.data_mix
    _seed(args.seed)

    model = build_framework(config)
    load_checkpoint(model, checkpoint)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    if not hasattr(model, "residual_world") or not hasattr(model.residual_world, "action_resamplers"):
        raise TypeError("attention sweep requires QwenResidualWorldPrefill")

    dataset = get_vla_dataset(config.datasets.vla_data, mode="val", seed=args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    examples = next(iter(loader))
    batch = model._prepare_batch(examples, training=False)
    max_horizon = args.max_horizon or int(model.residual_world.config.absorbing_horizon)
    horizons = np.arange(1, max_horizon + 1, dtype=np.int64)

    with torch.inference_mode():
        semantic, semantic_mask = model._encode_vlm(batch.vlm_images, batch.instructions)
        current = model._encode_vision(batch.world_images)
        prefill = model.residual_world.prefill(semantic, current, semantic_mask)
        layout = model.residual_world._flatten_features(current)
        if layout.spatial_shape is None:
            raise ValueError("attention visualization requires spatial vision features")

        residual_current, residual_self, action_residual, change_maps = [], [], [], []
        with WorldAttentionRecorder(model.residual_world) as recorder:
            for index, horizon in enumerate(horizons):
                recorder.clear()
                requested = torch.tensor([int(horizon)], device=device, dtype=torch.long)
                output = model.residual_world.reason(current, requested, prefill)
                snapshot = recorder.snapshot()
                residual_current.append(snapshot.residual_to_current.numpy())
                residual_self.append(snapshot.residual_to_residual.numpy())
                action_residual.append(snapshot.action_to_residual.numpy())
                change = output["task_change_map"][0].float().cpu().numpy()
                change_maps.append(change[None] if change.ndim == 2 else change)
                if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(horizons):
                    print(f"attention horizon {horizon}/{max_horizon}", flush=True)

    residual_current = spatialize(np.stack(residual_current), layout.spatial_shape, layout.view_count)
    residual_self = spatialize(np.stack(residual_self), layout.spatial_shape, layout.view_count)
    action_residual = spatialize(np.stack(action_residual), layout.spatial_shape, layout.view_count)
    change_maps = np.stack(change_maps)
    if args.view_index >= change_maps.shape[1]:
        raise IndexError(f"view_index {args.view_index} is invalid for {change_maps.shape[1]} views")

    output_dir = args.output_dir or (
        run_directory(checkpoint) / "evaluation" / checkpoint.stem.removesuffix("_pytorch_model")
        / f"attention_horizon_1_{max_horizon}"
    )
    current_image = np.asarray(batch.world_images[0][args.view_index])
    save_attention_outputs(
        output_dir, horizons, current_image, residual_current, residual_self, action_residual,
        change_maps, int(model.action_horizon), args.view_index,
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "instruction": batch.instructions[0],
        "horizon_min": 1,
        "horizon_max": max_horizon,
        "action_horizon": int(model.action_horizon),
        "absorbing_horizon": int(model.residual_world.config.absorbing_horizon),
        "view_index": args.view_index,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"attention sweep saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-mix", default="libero_residual_world_all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-horizon", type=int, default=0, help="0 uses the absorbing horizon")
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
