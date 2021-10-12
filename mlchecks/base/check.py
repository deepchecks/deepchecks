import abc
from typing import Callable, Dict, List, Any
from copy import deepcopy

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck', 'Decidable']


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
    params: Dict

    def __init__(self, **params):
        self.params = params


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


class Decidable(object):
    """
    Decidable is a utility class which gives the option to combine a check and a decision function to be used together

    WARNING: This class should always come first in the inheritance

    Example of usage:
    ```
    class MyCheck(Decidable, SingleDatasetBaseCheck):
        # run function signature is inherited from the check class
        def run(self, dataset, model=None) -> CheckResult:
            # Parameters are automatically sets on params property
            param1 = self.params.get('param1')
            # Do stuff...
            value, html = x, y
            return CheckResult(value, display={'text/html': html})

    my_check = MyCheck(param1='foo').decider(threshold(10))
    # Execute the run function and pass result to decide function
    my_check.decide(my_check.run())
    ```
    """
    _deciders: List[Callable] = []

    def __init__(self, deciders: List[Callable] = None, **params):
        self._deciders = deciders or []
        super().__init__(**params)

    def decide(self, result: CheckResult) -> bool:
        return all((x(result.value) for x in self._deciders))

    def decider(self, decider: Callable[[Any], bool]):
        new_copy = deepcopy(self)
        new_copy._deciders.append(decider)
        return new_copy
