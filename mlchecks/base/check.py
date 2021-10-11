import abc
from typing import Dict, Any

__all__ = ['CheckResult', 'Checkable']


class CheckResult:
    value: Any
    display: Dict

    def __init__(self, value, display=None):
        """
        Args:
            value (Any):
            display (Dict): Dictionary with formatters for display. possible
            foramtters are: 'text/html', 'image/png'
        """
        self.value = value
        self.display = display

    def _repr_mimebundle_(self, include, exclude):
        if not self.display:
            return None
        data = self.display
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data

    def __repr__(self):
        return self.value


class Checkable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, model=None, train_data=None, validation_data=None) -> CheckResult:
        pass

#
# class Decidable(Checkable):
#     """
#     Check is a utility class which gives the option to combine a check and a decision function
#     to be used together
#     """
#     deciders: List[Callable]
#
#     def __init__(self, deciders: List[Callable]=None, **check_params):
#         self.deciders = deciders or []
#         self.check_params = check_params
#
#     def run_and_decide(self, model=None, train_data=None, validation_data=None)
#     -> Tuple[CheckResult, bool]:
#         result = self.run(model=model, train_data=train_data, validation_data=validation_data)
#         decisions = [x(result.value) for x in self.deciders] or None
#         return result, decisions
#
#     def decider(self, decider: Callable[[Any], bool]):
#         deciders = [*self.deciders, decider]
#         return self.__class__(deciders=deciders, **self.check_params)
