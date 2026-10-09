"""LIBERO server-side episode state, kept outside the framework and wire payload."""


class EpisodePolicy:
    def __init__(self, framework, max_sessions=16):
        self.framework = framework
        self.sessions = {}
        self.max_sessions = max_sessions

    def predict_action(self, examples, *, ttt_session=None, ttt_episode=None, step=None, **kwargs):
        if not getattr(self.framework, "ttt_enabled", False):
            return self.framework.predict_action(examples=examples, **kwargs)
        if not isinstance(ttt_session, str) or not isinstance(ttt_episode, int) or step is None:
            raise ValueError("V3 requires the LIBERO_World client with session, episode and actual frame step")
        if len(examples) != 1:
            raise ValueError("LIBERO TTT server expects one environment per session")
        previous = self.sessions.get(ttt_session)
        if previous is None and len(self.sessions) >= self.max_sessions:
            raise ValueError("TTT session limit reached; restart the evaluation server")
        if previous is not None and ttt_episode < previous[0]:
            raise ValueError("stale episode request")
        state = previous[1] if previous is not None and previous[0] == ttt_episode else None
        output = self.framework.predict_action(examples=examples, step=step, state=state, **kwargs)
        output = dict(output)
        self.sessions[ttt_session] = (ttt_episode, output.pop("ttt_state"))
        return output
