"""Training-batch feedback diagnostics for the LIBERO TTT experiment."""

import torch


class TTTFeedbackEvalMixin:
    def eval_action_model(self, step_metrics=None):
        framework = self.accelerator.unwrap_model(self.model)
        if not getattr(framework, "ttt_enabled", False):
            return super().eval_action_model(step_metrics)

        # Random triplets have no persistent online episode state. Evaluate their
        # causal feedback objective, rather than invoking the stateful action API.
        # These are training-batch diagnostics, not held-out validation scores.
        examples = self._get_next_batch()
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad(), torch.autocast(
                framework.device.type, dtype=torch.bfloat16,
                enabled=framework.device.type == "cuda",
            ):
                output = framework.forward_ttt(examples)
            names = [name for name, value in output.items()
                     if name.startswith("ttt_") and isinstance(value, torch.Tensor)
                     and value.numel() == 1]
            values = torch.stack([output[name].detach().float().reshape(()) for name in names])
            values = self.accelerator.reduce(values, reduction="mean")
        finally:
            self.model.train(was_training)

        metrics = {} if step_metrics is None else step_metrics
        if self.accelerator.is_main_process:
            metrics.update({f"train_feedback/{name}": value.item()
                            for name, value in zip(names, values)})
        self.accelerator.wait_for_everyone()
        return metrics
