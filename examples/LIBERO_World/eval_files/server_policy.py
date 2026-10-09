"""Existing policy server with LIBERO-specific, explicit TTT episode state."""
import logging
from unittest.mock import patch

from deployment.model_server import server_policy
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from examples.LIBERO_World.eval_files.ttt_policy import EpisodePolicy


class WorldPolicyServerWrapper(PolicyServerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ttt_enabled = bool(getattr(self._framework, "ttt_enabled", False))
        if self._ttt_enabled:
            self._framework = EpisodePolicy(self._framework)

    @property
    def metadata(self):
        return dict(super().metadata, ttt_enabled=self._ttt_enabled)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = server_policy.build_argparser().parse_args()
    with patch.object(server_policy, "PolicyServerWrapper", WorldPolicyServerWrapper):
        server_policy.main(args)
