"""Module containing all the base classes for checks."""
import abc
import enum
import re
from collections import OrderedDict
from typing import Dict
from typing import Any, Callable, List, Union

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainValidationBaseCheck',
           'ModelOnlyBaseCheck', 'ConditionResult', 'ConditionCategory']

import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt

from mlchecks.string_utils import split_camel_case
from mlchecks.utils import MLChecksValueError


class Condition:
    """Contain condition attributes."""

    name: str
    function: Callable
    params: Dict

    def __init__(self, name: str, function: Callable, params: Dict):
        if not isinstance(function, Callable):
            raise MLChecksValueError(f'Condition must be a function `(Any) -> Union[ConditionResult, bool]`, '
                                     f'but got: {type(function).__name__}')
        if not isinstance(name, str):
            raise MLChecksValueError(f'Condition name must be of type str but got: {type(name).__name__}')
        self.name = name
        self.function = function
        self.params = params


class ConditionCategory(enum.Enum):
    """Condition result category. indicates whether the result should fail the suite."""

    FAIL = 'FAIL'
    INSIGHT = 'INSIGHT'


class ConditionResult:
    """Contain result of a condition function."""

    is_pass: bool
    description: str
    category: ConditionCategory
    actual: str

    def __init__(self, is_pass: bool, actual: str = '',
                 category: ConditionCategory = ConditionCategory.FAIL):
        """Initialize condition result.

        Args:
            is_pass (bool): Whether the condition functions passed the given value or not.
            actual (str): What actual value was found.
            category (ConditionCategory): The category to which the condition result belongs.
        """
        self.is_pass = is_pass
        self.actual = actual
        self.category = category

    def set_description(self, description: str):
        """Set description to be displayed in table.

        Args:
            description (str): Description of the condition to be displayed.
        """
        self.description = description

    def get_icon(self):
        if self.is_pass:
            return '\U0001F389'
        elif self.category == ConditionCategory.FAIL:
            return '\U0001F631'
        else:
            return '\U0001F937'


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
    condition_results: List[ConditionResult]

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
                raise MLChecksValueError(f'Can\'t display item of type: {type(item)}')

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

    def set_condition_results(self, results: List[ConditionResult]):
        """Set the conditions results for current check result."""
        self.conditions_results = results


class BaseCheck(metaclass=abc.ABCMeta):
    """Base class for check."""

    _conditions: OrderedDict
    _conditions_index: int

    def __init__(self):
        """Init base check parameters to pass to be used in the implementing check."""
        self._conditions = OrderedDict()
        self._conditions_index = 0

    def conditions_decision(self, result: CheckResult) -> List[ConditionResult]:
        """Run conditions on given result."""
        results = []
        condition: Condition
        for condition in self._conditions.values():
            output = condition.function(result.value, **condition.params)
            if isinstance(output, bool):
                output = ConditionResult(output)
            elif not isinstance(output, ConditionResult):
                raise MLChecksValueError(f'Invalid return type from condition {condition.name}, got: {type(output)}')
            output.set_description(condition.name)
            results.append(output)
        return results

    def add_condition(self, name: str, condition_func: Callable[[Any], Union[ConditionResult, bool]], **params):
        """Add new condition function to the check.

        Args:
            name (str): Name of the condition. should explain the condition action and parameters
            condition_func (Callable[[Any], Union[List[ConditionResult], bool]]): Function which gets the value of the
                check and returns object of List[ConditionResult] or boolean.
            params: Additional parameters to pass when calling the condition function.
        """
        cond = Condition(name, condition_func, params)
        self._conditions[self._conditions_index] = cond
        self._conditions_index += 1
        return self

    def __repr__(self, tabs=0):
        """Representation of check as string.

        Args:
            tabs (int): number of tabs to shift by the output
        """
        tab_chr = '\t'
        params = self.params()
        params = f'({params})' if params else ''
        check_str = f'{tab_chr * tabs}{self.__class__.__name__}{params}'
        if self._conditions:
            conditions_str = ''.join([f'\n{tab_chr * (tabs + 2)}{i}: {s.name}' for i, s in self._conditions.items()])
            return f'{check_str}\n{tab_chr * (tabs + 1)}Conditions: [{conditions_str}\n{tab_chr * (tabs + 1)}]'
        else:
            return check_str

    def params(self):
        return ''

    def clean_conditions(self):
        """Remove all conditions from this check instance."""
        self._conditions.clear()
        self._conditions_index = 0

    def remove_condition(self, index: int):
        """Remove given condition by index.

        Args:
            index (int): index of condtion to remove
        """
        if index not in self._conditions:
            raise MLChecksValueError(f'Index {index} of conditions does not exists')
        self._conditions.pop(index)


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
