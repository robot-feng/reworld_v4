"""Reuse the standard StarVLA trainer with causal TTT periodic diagnostics."""

import argparse
from unittest.mock import patch

from omegaconf import OmegaConf

from examples.LIBERO_World.train_files.ttt_trainer import TTTFeedbackEvalMixin
from starVLA.training import train_starvla


class TTTTrainer(TTTFeedbackEvalMixin, train_starvla.VLATrainer):
    def _train_step(self, batch_vla, batch_vlm=None):
        framework = self.accelerator.unwrap_model(self.model)
        framework.training_step = self.completed_steps
        metrics = super()._train_step(batch_vla, batch_vlm)
        if getattr(framework, "joint_training", False):
            metrics["ttt_aux_weight"] = framework.auxiliary_loss_weight()
        return metrics


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
