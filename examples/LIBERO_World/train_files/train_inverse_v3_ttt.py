"""Reuse the standard StarVLA trainer with causal TTT periodic diagnostics."""

import argparse
import torch
from unittest.mock import patch

from omegaconf import OmegaConf

from examples.LIBERO_World.train_files.ttt_trainer import TTTFeedbackEvalMixin
from starVLA.training import train_starvla


class TTTTrainer(TTTFeedbackEvalMixin, train_starvla.VLATrainer):
    def prepare_training(self):
        result = super().prepare_training()
        # DeepSpeed finalizes rank count during prepare(). Refresh the displayed
        # effective batch now so 4 ranks x 16 x 2 reports 128, not the stale 64.
        world_size = max(
            train_starvla.dist.get_world_size() if train_starvla.dist.is_initialized() else 1,
            int(self.accelerator.num_processes),
        )
        self.total_batch_size = (
            self.config.datasets.vla_data.per_device_batch_size
            * world_size * self.accelerator.gradient_accumulation_steps
        )
        return result

    def _train_step(self, batch_vla, batch_vlm=None):
        framework = self.accelerator.unwrap_model(self.model)
        framework.training_step = self.completed_steps
        gas = self.accelerator.gradient_accumulation_steps
        if not hasattr(self, "_v3_micro_step"):
            self._v3_micro_step = 0
        with self.accelerator.accumulate(self.model):
            if self._v3_micro_step % gas == 0:
                self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.model.forward(batch_vla)
                total_loss, loss_metrics = train_starvla.extract_model_losses(output)
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
        self._v3_micro_step += 1
        metrics = {name: value.item() for name, value in loss_metrics.items()}
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
