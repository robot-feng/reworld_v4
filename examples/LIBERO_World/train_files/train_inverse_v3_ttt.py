"""Reuse the standard StarVLA trainer with causal TTT periodic diagnostics."""

import argparse
from unittest.mock import patch

from omegaconf import OmegaConf

from examples.LIBERO_World.train_files.ttt_trainer import TTTFeedbackEvalMixin
from starVLA.training import train_starvla


class TTTTrainer(TTTFeedbackEvalMixin, train_starvla.VLATrainer):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist(train_starvla.normalize_dotlist_args(overrides)),
    )
    cfg = train_starvla.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    if cfg.is_debug and train_starvla.dist.is_initialized() and train_starvla.dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        debugpy.wait_for_client()
    with patch.object(train_starvla, "VLATrainer", TTTTrainer):
        train_starvla.main(cfg)


if __name__ == "__main__":
    main()
