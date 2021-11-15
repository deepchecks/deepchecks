"""Module containing all the base classes for checks."""
import abc
import re
from typing import Any, Callable, List, Union

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck']

import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt

from deepchecks.string_utils import split_camel_case
from deepchecks.utils import DeepchecksValueError


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
    display: List[Union[Callable, str, pd.DataFrame]]

    def __init__(self, value, header: str = None, check=None, display: Any = None):
        """Init check result.

        Args:
            value (Any): Value calculated by check. Can be used to decide if decidable check passed.
            header (str): Header to be displayed in python notebook.
            check (Callable): The check function which created this result. Used to extract the summary to be
            displayed in notebook.
            display (Callable): Function which is used for custom display.
        """
        self.value = value
        self.header = header or (check and split_camel_case(check.__name__)) or None
        self.check = check

        if display is not None and not isinstance(display, List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Callable)):
                raise DeepchecksValueError(f'Can\'t display item of type: {type(item)}')

    def _ipython_display_(self):
        if self.header:
            display_html(f'<h4>{self.header}</h4>', raw=True)
        if self.check and '__doc__' in dir(self.check):
            docs = self.check.__doc__
            # Take first non-whitespace line.
            summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')
            display_html(f'<p>{summary}</p>', raw=True)

        for item in self.display:
            if isinstance(item, pd.DataFrame):
                # Align everything to the left
                try:
                    df_styler = item.style
                    df_styler.set_table_styles([dict(selector='th,td', props=[('text-align', 'left')])])
                    display_html(df_styler.render(), raw=True)
                # Because of MLC-154. Dataframe with Multi-index or non unique indices does not have a style
                # attribute, hence we need to display as a regular pd html format.
                except ValueError:
                    display_html(item.to_html())
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


class BaseCheck(metaclass=abc.ABCMeta):
    """Base class for check."""

    def __repr__(self):
        """Representation of check as string."""
        return f'{self.__class__.__name__}'


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
#             raise DeepchecksValueError(f'Validator must be a function in signature `(CheckResult) -> bool`,'
#                                       'but got: {type(decider).__name__}')
#         new_copy = deepcopy(self)
#         new_copy._validators.append(validator)
#         return new_copy
