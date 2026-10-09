"""Derive a memory-only LIBERO run from the actual trained V2 configuration."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf


def prepare(base, checkpoint, output):
    cfg = OmegaConf.load(base)
    if cfg.framework.name != "QwenResidualWorldInverseV2":
        raise ValueError("base config must describe the trained QwenResidualWorldInverseV2 checkpoint")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    action = cfg.framework.action_model
    horizon = action.get("action_horizon")
    if horizon is None:
        horizon = int(action.future_action_window_size) + 1
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("V2 action_horizon must be a positive integer")
    horizon = int(horizon)
    data = cfg.datasets.vla_data
    max_horizon = int(data.get("trajectory_max_horizon", data.get("future_target_max_horizon", 500)))
    if max_horizon < 2 * horizon:
        raise ValueError("trajectory_max_horizon must cover two action horizons for TTT training")
    if int(cfg.framework.residual_world.get("absorbing_horizon", 500)) < horizon:
        raise ValueError("absorbing_horizon must cover the action horizon for TTT feedback")
    template = OmegaConf.load(Path(__file__).with_name("starvla_qwen_residual_world_inverse_v3.yaml"))
    cfg.framework.name = "QwenResidualWorldInverseV3"
    cfg.framework.ttt = template.framework.ttt
    cfg.framework.inference_horizon = horizon
    # Preserve architecture, views, data paths, normalization and action setup.
    cfg.datasets.vla_data.trajectory_fixed_mid = horizon
    cfg.datasets.vla_data.trajectory_fixed_future = 2 * horizon
    cfg.trainer.learning_rate = template.trainer.learning_rate
    cfg.trainer.freeze_modules = template.trainer.freeze_modules
    cfg.trainer.pretrained_checkpoint = str(checkpoint)
    cfg.trainer.reload_modules = template.trainer.reload_modules
    cfg.trainer.is_resume = False
    cfg.run_id = str(cfg.get("run_id", "libero")) + "_v3_ttt"
    cfg.pop("output_dir", None)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.base, args.checkpoint, args.output)
