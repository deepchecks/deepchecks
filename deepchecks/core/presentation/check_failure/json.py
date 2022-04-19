import typing as t

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import JsonSerializer


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(JsonSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value
    
    def serialize(self, **kwargs) -> t.Dict[t.Any, t.Any]:
        return {
            'header': self.value.header,
            'check': self.value.check.metadata(),
        }