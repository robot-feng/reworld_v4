"""Official LIBERO actions/chunk scheduling plus V3 episode/frame metadata."""
from uuid import uuid4
from numbers import Integral

from examples.LIBERO.eval_files.model2libero_interface import ModelClient as OfficialModelClient


class ModelClient(OfficialModelClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ttt_session = uuid4().hex
        self._ttt_episode = 0
        self._frame_step = 0
        self._send_action = self.client.predict_action
        self.client.predict_action = self._predict_with_state

    def reset(self, task_description):
        super().reset(task_description)
        self._ttt_episode += 1
        self._frame_step = 0

    def step(self, example, step=0, **kwargs):
        # The official method may reset on an instruction change; do it first
        # so that reset cannot overwrite this observation's frame index.
        if example.get("lang") != self.task_description:
            self.reset(example.get("lang"))
        self._frame_step = step
        return super().step(example, step=step, **kwargs)

    def _predict_with_state(self, request):
        if self._server_metadata.get("ttt_enabled", False):
            for name in ("horizon", "long_horizon"):
                value = request.get(name)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, Integral)
                    or value < 1 or value % self.action_chunk_size
                ):
                    raise ValueError(f"TTT {name} must be a positive multiple of action_chunk_size; "
                                     "LIBERO only sends observations at chunk boundaries")
            request = dict(request, ttt_session=self._ttt_session,
                           ttt_episode=self._ttt_episode, step=self._frame_step)
        return self._send_action(request)


__all__ = ["ModelClient"]
