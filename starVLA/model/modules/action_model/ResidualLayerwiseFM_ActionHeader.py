"""Residual-world-specific layer-wise flow-matching action head.

The upstream ``LayerwiseFM_ActionHeader`` remains untouched. This subclass
only owns the residual world's interleaved layer execution.
"""

from __future__ import annotations

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


class ResidualLayerwiseFlowmatchingActionHead(
    LayerwiseFlowmatchingActionHead
):
    """Layer-wise FM head with alternating world cross/action self-attention."""

    def _action_tokens(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        state_features: torch.Tensor | None,
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        if self.config.add_pos_embed:
            positions = torch.arange(
                action_features.shape[1],
                device=actions.device,
            )
            action_features = (
                action_features
                + self.position_embedding(positions).unsqueeze(0)
            )

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            actions.shape[0],
            -1,
            -1,
        )
        parts = (
            (state_features, future_tokens, action_features)
            if state_features is not None
            else (future_tokens, action_features)
        )
        return torch.cat(parts, dim=1)

    def _run_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        contexts: list[torch.Tensor],
        timesteps: torch.Tensor,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        blocks = self.model.transformer_blocks
        if len(contexts) != len(blocks):
            raise ValueError(
                f"Expected {len(blocks)} layer-wise contexts, "
                f"got {len(contexts)}"
            )

        temb = self.model.timestep_encoder(timesteps)
        interleaved = bool(
            getattr(self.model.config, "interleave_self_attention", False)
        )
        for index, block in enumerate(blocks):
            self_attention = interleaved and index % 2 == 1
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=(
                    None if self_attention else contexts[index]
                ),
                encoder_attention_mask=(
                    None if self_attention else encoder_attention_mask
                ),
                temb=temb,
            )
        return hidden_states

    def _predict_velocity(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        state_features: torch.Tensor | None,
        contexts: list[torch.Tensor],
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        hidden_states = self._action_tokens(
            actions,
            timesteps,
            state_features,
        )
        hidden_states = self._run_transformer_blocks(
            hidden_states,
            contexts,
            timesteps,
            encoder_attention_mask,
        )
        return self.action_decoder(hidden_states)[:, -actions.shape[1] :]

    def forward(
        self,
        vl_embs_list: list[torch.Tensor],
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        noise = torch.randn_like(actions)
        time = self.sample_time(
            actions.shape[0],
            actions.device,
            actions.dtype,
        )[:, None, None]
        noisy_actions = (1 - time) * noise + time * actions
        timesteps = (
            time[:, 0, 0] * self.num_timestep_buckets
        ).long()
        state_features = (
            self.state_encoder(state) if state is not None else None
        )
        velocity = self._predict_velocity(
            noisy_actions,
            timesteps,
            state_features,
            vl_embs_list,
            encoder_attention_mask,
        )
        return (velocity - (actions - noise)).square().mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor | None = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs_list[0].shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=vl_embs_list[0].device,
            dtype=vl_embs_list[0].dtype,
        )
        state_features = (
            self.state_encoder(state) if state is not None else None
        )
        step_size = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            timesteps = torch.full(
                (batch_size,),
                int(
                    step
                    / self.num_inference_timesteps
                    * self.num_timestep_buckets
                ),
                device=actions.device,
                dtype=torch.long,
            )
            actions = actions + step_size * self._predict_velocity(
                actions,
                timesteps,
                state_features,
                vl_embs_list,
                encoder_attention_mask,
            )
        return actions


def get_action_model(
    config=None,
) -> ResidualLayerwiseFlowmatchingActionHead:
    return ResidualLayerwiseFlowmatchingActionHead(global_config=config)


def _smoke_test() -> None:
    """Verify alternating world cross-attention/action self-attention."""

    from types import SimpleNamespace

    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used_context: bool | None = None
            self.used_mask: bool | None = None

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            temb=None,
        ):
            del temb
            self.used_context = encoder_hidden_states is not None
            self.used_mask = encoder_attention_mask is not None
            return hidden_states + 1

    head = ResidualLayerwiseFlowmatchingActionHead.__new__(
        ResidualLayerwiseFlowmatchingActionHead
    )
    nn.Module.__init__(head)
    blocks = nn.ModuleList([_Block() for _ in range(4)])
    head.model = nn.Module()
    head.model.transformer_blocks = blocks
    head.model.timestep_encoder = nn.Identity()
    head.model.config = SimpleNamespace(
        interleave_self_attention=True
    )

    hidden = torch.zeros(2, 3, 8)
    contexts = [torch.randn(2, 5, 8) for _ in blocks]
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = head._run_transformer_blocks(
        hidden,
        contexts,
        torch.zeros(2, dtype=torch.long),
        encoder_attention_mask=mask,
    )
    assert torch.equal(output, torch.full_like(hidden, 4))
    assert [block.used_context for block in blocks] == [
        True,
        False,
        True,
        False,
    ]
    assert [block.used_mask for block in blocks] == [
        True,
        False,
        True,
        False,
    ]
    print("ResidualLayerwiseFlowmatchingActionHead smoke test passed")


if __name__ == "__main__":
    _smoke_test()
