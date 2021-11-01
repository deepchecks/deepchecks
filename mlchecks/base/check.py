"""Module containing all the base classes for checks."""
import abc
import enum
import re
from collections import OrderedDict
from typing import Dict, Any, Callable, List, Union

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck', 'ValidateResult', 'ValidateCategory']

import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt

from mlchecks.string_utils import underscore_to_capitalize
from mlchecks.utils import MLChecksValueError


class CheckResult:
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Attributes:
        value (Any): Value calculated by check. Can be used to decide if decidable check passed.
        display (Dict): Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    """

    value: Any
    header: str
    check: Callable
    display: List[Union[Callable, str, pd.DataFrame]]

    def __init__(self, value, header: str = None, check: Callable = None, display: Any = None):
        """Init check result.

        Args:
            value (Any): Value calculated by check. Can be used to decide if decidable check passed.
            header (str): Header to be displayed in python notebook.
            check (Callable): The check function which created this result. Used to extract the summary to be
            displayed in notebook.
            display (Callable): Function which is used for custom display.
        """
        if check is not None and not isinstance(check, Callable):
            raise MLChecksValueError('`check` parameter of CheckResult must be callable')
        self.value = value
        self.header = header or (check and underscore_to_capitalize(check.__name__)) or None
        self.check = check

        if display is not None and not isinstance(display, List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Callable)):
                raise MLChecksValueError(f'Can\'t display item of type: {type(item)}')

    def _ipython_display_(self):
        if self.header:
            display_html(f'<h4>{self.header}</h4>', raw=True)
        if self.check:
            docs = self.check.__doc__
            # Take first non-whitespace line.
            summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')
            display_html(f'<p>{summary}</p>', raw=True)

        for item in self.display:
            if isinstance(item, pd.DataFrame):
                # Align everything to the left
                df_styler = item.style
                df_styler.set_table_styles([dict(selector='th,td', props=[('text-align', 'left')])])
                display_html(df_styler.render(), raw=True)
            elif isinstance(item, str):
                display_html(item, raw=True)
            elif isinstance(item, Callable):
                item()
                plt.show()
            else:
                raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            display_html('<p><b>&#x2713;</b> Nothing found</p>', raw=True)

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.value.__repr__()


class ValidateCategory(enum.Enum):
    Error = 'Error'
    Insight = 'Insight'


class ValidateResult:
    """Contain result of a validation function."""

    is_pass: bool
    category: ValidateCategory
    expected: str
    actual: str

    def __init__(self, is_pass: bool, expected: str = None, actual: str = None,
                 category: ValidateCategory = ValidateCategory.Insight):
        self.is_pass = is_pass
        self.expected = expected
        self.actual = actual
        self.category = category


class BaseCheck(metaclass=abc.ABCMeta):
    """Base class for check."""

    params: Dict
    _validators: OrderedDict

    def __init__(self, **kwargs):
        """Init base check parameters to pass to be used in the implementing check."""
        self._validators = OrderedDict()
        self.params = kwargs

    def validate(self, result: CheckResult) -> Dict[str, ValidateResult]:
        results = {}
        for name, validator in self._validators.items():
            output = validator(result.value)
            if isinstance(output, ValidateResult):
                results[name] = output
            elif isinstance(output, bool):
                results[name] = ValidateResult(output)
            else:
                raise MLChecksValueError(f'Invalid return type from validation {name}, got: {type(output)}')
        return results

    def add_validator(self, name: str, validator: Callable[[CheckResult], ValidateResult]):
        if not isinstance(validator, Callable):
            raise MLChecksValueError(f'Validator must be a function in signature `(CheckResult) -> ValidateResult`,'
                                     f'but got: {type(validator).__name__}')
        if not isinstance(name, str):
            raise MLChecksValueError(f'validator name must be of type str but got: {type(name).__name__}')
        self._validators[name] = validator

    def __repr__(self, tabs=0):
        """Representation of check as string."""
        tabs_str = '\t' * tabs
        validator_tabs = '\t' * (tabs + 1)
        check_str = f'{tabs_str}{self.__class__.__name__}({self.params})'
        if self._validators:
            validator_str = ''.join([f'\n{validator_tabs}{s}' for s in self._validators.keys()])
            return f'{check_str} Validators: [{validator_str}\n{tabs_str}]'
        else:
            return check_str

    def update_param(self, param: str, value):
        self.params[param] = value

    def remove_param(self, param: str):
        self.params.pop(param)

    def clean_validators(self):
        self._validators.clear()

    def remove_validator(self, name: str):
        self._validators.pop(name)


class SingleDatasetBaseCheck(BaseCheck):
    """Parent class for checks that only use one dataset."""

    @abc.abstractmethod
    def run(self, dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class CompareDatasetsBaseCheck(BaseCheck):
    """Parent class for checks that compare between two datasets."""

    @abc.abstractmethod
    def run(self, dataset, baseline_dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class TrainValidationBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and validation dataset for model training and validation.
    """

    @abc.abstractmethod
    def run(self, train_dataset, validation_dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    @abc.abstractmethod
    def run(self, model) -> CheckResult:
        """Define run signature."""
        pass



# class Validatable(metaclass=abc.ABCMeta):
#     """Validatable combines a check with validate functions to be used together.
#
#     Example of usage:
#     ```
#     class MyCheck(Validatable, SingleDatasetBaseCheck):
#         # run function signaute is inherited from the check class
#         def run(self, dataset, model=None) -> CheckResult:
#             # Parameters are automaticlly sets on params property
#             param1 = self.params.get('param1')
#             # Do stuff...
#             value, html = x, y
#             return CheckResult(value, display=html)
#
#         def validate_x_larger_than(value: int) -> bool
#             def validate(result: CheckResult):
#                 return result.value > value
#             return self.add_validator(validate)
#
#     my_check = MyCheck(param1='foo').validate_x_larger_than(400)
#     # Execute the run function and pass result to decide function
#     my_check.validate(my_check.run())
#     ```
#     """
#
#     _validators: OrderedDict[str, Callable]
#
#     def __init__(self, *params, **kwargs):
#         self._validators = OrderedDict()
#         super().__init__(*params, **kwargs)
#
#     def validate(self, result: CheckResult) -> Dict[str, ValidateResult]:
#         return {name: validator(result.value) for name, validator in self._validators.items()}
#
#     def add_validator(self, name: str, validator: Callable[[CheckResult], ValidateResult]):
#         if not isinstance(validator, Callable):
#             raise MLChecksValueError(f'Validator must be a function in signature `(CheckResult) -> ValidateResult`,'
#                                      f'but got: {type(validator).__name__}')
#         if not isinstance(name, str):
#             raise MLChecksValueError(f'validator name must be of type str but got: {type(name).__name__}')
#         self._validators[name] = validator
