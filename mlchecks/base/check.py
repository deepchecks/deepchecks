import abc
from typing import Callable, Dict, List, Any, Tuple

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck']


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
            data = {k:v for (k,v) in data.items() if k in include}
        if exclude:
            data = {k:v for (k,v) in data.items() if k not in exclude}
        return data

    def __repr__(self):
        return self.value.__repr__()


class BaseCheck(metaclass=abc.ABCMeta):
    pass


class SingleDatasetBaseCheck(BaseCheck):
    """
    Parent class for checks that only use one dataset
    """
    @abc.abstractmethod
    def run(self, dataset, model=None) -> CheckResult:
        pass


class CompareDatasetsBaseCheck(BaseCheck):
    """
    Parent class for checks that compare between two datasets
    """
    @abc.abstractmethod
    def run(self, dataset, compared_dataset, model=None) -> CheckResult:
        pass


class TrainValidationBaseCheck(BaseCheck):
    """
    Parent class for checks that compare two datasets - train dataset and validation dataset
    for model training and validation
    """
    @abc.abstractmethod
    def run(self, train_dataset, validation_dataset, model=None) -> CheckResult:
        pass


class ModelOnlyBaseCheck(BaseCheck):
    """
    Parent class for checks that only use a model and no datasets
    """
    @abc.abstractmethod
    def run(self, model) -> CheckResult:
        pass

#
# class Decidable(Checkable):
#     """
#     Check is a utility class which gives the option to combine a check and a decision function to be used together
#     """
#     deciders: List[Callable]
#
#     def __init__(self, deciders: List[Callable]=None, **check_params):
#         self.deciders = deciders or []
#         self.check_params = check_params
#
#     def run_and_decide(self, model=None, train_data=None, validation_data=None) -> Tuple[CheckResult, bool]:
#         result = self.run(model=model, train_data=train_data, validation_data=validation_data)
#         decisions = [x(result.value) for x in self.deciders] or None
#         return result, decisions
#
#     def decider(self, decider: Callable[[Any], bool]):
#         deciders = [*self.deciders, decider]
#         return self.__class__(deciders=deciders, **self.check_params)
