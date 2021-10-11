from typing import Callable, Dict, List, Any, Tuple

from mlchecks.base.check import Checkable

__all__ = ['CheckSuite']


class CheckSuite:
    checks: List[Checkable]

    def __init__(self, *checks):
        for check in checks:
            if not isinstance(check, Checkable):
                raise Exception(f'CheckSuite receives only `Checkable` objects but got: {check.__class__.__name__}')
        self.checks = checks

    def run(self, model=None, train_data=None, validation_data=None):
        return [check.run(model=model, train_data=train_data, validation_data=validation_data)
                for check in self.checks]

    def run_and_decide(self, model=None, train_data=None, validation_data=None):
        return [check.run_and_decide(model=model, train_data=train_data, validation_data=validation_data)
                for check in self.checks]
