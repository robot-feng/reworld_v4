"""Small tensor-only metric set for residual world predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor


def _mean_square(values: Tensor) -> Tensor:
    return values.float().flatten(1).square().mean(dim=1)


def _prediction_metrics(
    current: Tensor,
    predicted_future: Tensor,
    target_future: Tensor,
    predicted_delta: Tensor,
    target_delta: Tensor,
) -> dict[str, Tensor]:
    epsilon = torch.finfo(torch.float32).eps
    delta_mse = _mean_square(predicted_delta - target_delta)
    target_delta_energy = _mean_square(target_delta)
    future_mse = _mean_square(predicted_future - target_future)
    copy_current_mse = _mean_square(current - target_future)
    target_future_energy = _mean_square(target_future)
    cosine = torch.nn.functional.cosine_similarity(
        predicted_delta.flatten(1),
        target_delta.flatten(1),
        dim=1,
        eps=1e-8,
    )
    return {
        "delta_mse": delta_mse,
        "delta_nmse": delta_mse / target_delta_energy.clamp_min(epsilon),
        "delta_cosine": cosine,
        "future_mse": future_mse,
        "future_nmse": future_mse / target_future_energy.clamp_min(epsilon),
        "gain_vs_copy_current": (
            copy_current_mse - future_mse
        ) / copy_current_mse.clamp_min(epsilon),
        "target_delta_rms": target_delta_energy.sqrt(),
    }


def batch_world_metrics(output: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return direct and, when available, self-forced metrics per sample."""
    required = (
        "current_vision_features",
        "predicted_future_vision_features",
        "target_vision_features",
        "predicted_feature_delta",
        "target_feature_delta",
    )
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError("world output is missing: " + ", ".join(missing))

    current = output["current_vision_features"].float()
    direct = _prediction_metrics(
        current,
        output["predicted_future_vision_features"].float(),
        output["target_vision_features"].float(),
        output["predicted_feature_delta"].float(),
        output["target_feature_delta"].float(),
    )
    expanded = (
        "target_mid_vision_features",
        "target_mid_feature_delta",
        "predicted_mid_vision_features",
        "predicted_mid_feature_delta",
        "predicted_rollout_vision_features",
        "predicted_rollout_feature_delta",
    )
    if not all(key in output for key in expanded):
        return direct

    stages = {
        "mid": _prediction_metrics(
            current,
            output["predicted_mid_vision_features"].float(),
            output["target_mid_vision_features"].float(),
            output["predicted_mid_feature_delta"].float(),
            output["target_mid_feature_delta"].float(),
        ),
        "direct_future": direct,
        "rollout_future": _prediction_metrics(
            current,
            output["predicted_rollout_vision_features"].float(),
            output["target_vision_features"].float(),
            output["predicted_rollout_feature_delta"].float(),
            output["target_feature_delta"].float(),
        ),
    }
    metrics = {f"{stage}_{name}": value for stage, values in stages.items() for name, value in values.items()}
    epsilon = torch.finfo(torch.float32).eps
    metrics["rollout_gain_vs_direct_future"] = (
        stages["direct_future"]["future_mse"] - stages["rollout_future"]["future_mse"]
    ) / stages["direct_future"]["future_mse"].clamp_min(epsilon)
    return metrics


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all/terminal/non-terminal groups without hidden weighting."""

    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups["terminal" if record["target_is_terminal"] else "non_terminal"].append(record)

    metric_names = tuple(
        key
        for key, value in records[0].items()
        if key not in {"sample_index", "dataset_name", "horizon_mid", "horizon", "target_is_terminal"}
        and isinstance(value, (int, float))
    )
    return {
        name: {
            "count": len(items),
            **{
                metric: sum(float(item[metric]) for item in items) / len(items)
                for metric in metric_names
            },
        }
        for name, items in groups.items()
    }


def _smoke_test() -> None:
    current = torch.zeros(2, 4, 3, 3)
    target = torch.ones_like(current)
    output = {
        "current_vision_features": current,
        "predicted_future_vision_features": target,
        "target_vision_features": target,
        "predicted_feature_delta": target,
        "target_feature_delta": target,
    }
    metrics = batch_world_metrics(output)
    assert torch.equal(metrics["delta_mse"], torch.zeros(2))
    assert torch.equal(metrics["gain_vs_copy_current"], torch.ones(2))

    output.update(
        target_mid_vision_features=target / 2,
        target_mid_feature_delta=target / 2,
        predicted_mid_vision_features=target / 2,
        predicted_mid_feature_delta=target / 2,
        predicted_rollout_vision_features=target,
        predicted_rollout_feature_delta=target,
    )
    metrics = batch_world_metrics(output)
    assert torch.equal(metrics["mid_future_mse"], torch.zeros(2))
    assert torch.equal(metrics["rollout_future_future_mse"], torch.zeros(2))
    print("LIBERO_World metrics smoke test passed")


if __name__ == "__main__":
    _smoke_test()
