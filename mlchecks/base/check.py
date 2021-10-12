"""Module containing base classes for all checks"""
import abc
from typing import Callable, Dict, List, Any
from copy import deepcopy

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck', 'Decidable']


class CheckResult:
    """Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Attributes:
        value (Any): Value calculated by check. Can be used to decide if decidable check passed.
        display (Dict): Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    """
    value: Any
    display: Dict

    def __init__(self, value, display=None):
        """
        Args:
            value (Any): Value calculated by check. Can be used to decide if decidable check passed.
            display (Dict): Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
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

    Example of usage:
    ```
    class MyCheck(Decidable, SingleDatasetBaseCheck):
        # run function signaute is inherited from the check class
        def run(self, dataset, model=None) -> CheckResult:
            # Parameters are automaticlly sets on params property
            param1 = self.params.get('param1')
            # Do stuff...
            value, html = x, y
            return CheckResult(value, display={'text/html': html})

    my_check = MyCheck(param1='foo').decider(threshold(10))
    # Execute the run function and pass result to decide function
    my_check.decide(my_check.run())
    ```
    """
    _deciders: List[Callable]

    def __init__(self, deciders: List[Callable] = None, **params):
        self._deciders = deciders or []
        super().__init__(**params)

    def decide(self, result: CheckResult) -> List[bool]:
        return [x(result.value) for x in self._deciders] or None

    def decider(self, decider: Callable[[Any], bool]):
        new_copy = deepcopy(self)
        new_copy._deciders.append(decider) # pylint: disable=protected-access
        return new_copy                    # we access this variable from within its own class after deep copy
