"""
Module containing all the base classes for checks
"""
import abc
from typing import Dict, Any

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck']


class CheckResult:
    """Class which returns from a check with result that can later be used for automatic pipelines and display value
    """
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


# class Validatable(metaclass=abc.ABCMeta):
#     """
#     Decidable is a utility class which gives the option to combine a check and a decision function to be used together
#
#     Example of usage:
#     ```
#     class MyCheck(Decidable, SingleDatasetBaseCheck):
#         # run function signaute is inherited from the check class
#         def run(self, dataset, model=None) -> CheckResult:
#             # Parameters are automaticlly sets on params property
#             param1 = self.params.get('param1')
#             # Do stuff...
#             value, html = x, y
#             return CheckResult(value, display={'text/html': html})
#
#         # Implement default decider
#         def default_decider(result: CheckResult, param=None, param2=None, param3=None) -> bool
#             # To stuff...
#             return True
#
#         # Implements "syntactic sugar" for decider function
#         def decide_on_param_2(param):
#             return self.decider({param2: param})
#
#     my_check = MyCheck(param1='foo').decider(param2=10)
#     my_check = MyCheck(param1='foo').decider(param='s', param2=10)
#     my_check = MyCheck(param1='foo').decide_on_param_2(10)
#     my_check = MyCheck(param1='foo').decider(lambda cr: cr.value > 0)
#     # Execute the run function and pass result to decide function
#     my_check.decide(my_check.run())
#     ```
#     """
#     _validators: List[Callable]
#
#     def __init__(self, **params):
#         self._validators = []
#         super().__init__(**params)
#
#     def validate(self, result: CheckResult) -> List[bool]:
#         decisions = []
#         for curr_validator in self._validators:
#             decisions.append(curr_validator(result))
#         return decisions or None
#
#     def add_validator(self, validator: Callable[[CheckResult], bool]):
#         if not not isinstance(validator, Callable):
#             raise MLChecksValueError(f'Validator must be a function in signature `(CheckResult) -> bool`,'
#                                       'but got: {type(decider).__name__}')
#         new_copy = deepcopy(self)
#         new_copy._validators.append(validator)
#         return new_copy
