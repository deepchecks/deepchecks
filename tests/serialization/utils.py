from deepchecks.core.checks import BaseCheck


class DummyCheck(BaseCheck):
    """Dummy check type for testing purpose."""

    def run(self, *args, **kwargs):
        raise NotImplementedError()
