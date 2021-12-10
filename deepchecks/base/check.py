# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing all the base classes for checks."""
import typing as t
import abc
import enum
import re
import typing
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps

import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler

from deepchecks.base import dataset
from deepchecks.base.display_pandas import display_dataframe
from deepchecks.utils.strings import split_camel_case
from deepchecks.utils.metrics import ModelType, infer_task_type
from deepchecks.utils.validation import ensure_not_empty_dataset
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


__all__ = [
    'CheckResult',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'CompareDatasetsBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ConditionResult',
    'ConditionCategory',
    'CheckFailure'
]


class Condition:
    """Contain condition attributes."""

    name: str
    function: t.Callable
    params: t.Dict

    def __init__(self, name: str, function: t.Callable, params: t.Dict):
        if not isinstance(function, t.Callable):
            raise DeepchecksValueError(f'Condition must be a function `(Any) -> Union[ConditionResult, bool]`, '
                                       f'but got: {type(function).__name__}')
        if not isinstance(name, str):
            raise DeepchecksValueError(f'Condition name must be of type str but got: {type(name).__name__}')
        self.name = name
        self.function = function
        self.params = params

    def __call__(self, *args, **kwargs) -> 'ConditionResult':
        result = t.cast(ConditionResult, self.function(*args, **kwargs))
        result.set_name(self.name)
        return result


class ConditionCategory(enum.Enum):
    """Condition result category. indicates whether the result should fail the suite."""

    FAIL = 'FAIL'
    WARN = 'WARN'


class ConditionResult:
    """Contain result of a condition function."""

    is_pass: bool
    category: ConditionCategory
    details: str
    name: str

    def __init__(self, is_pass: bool, details: str = '',
                 category: ConditionCategory = ConditionCategory.FAIL):
        """Initialize condition result.

        Args:
            is_pass (bool): Whether the condition functions passed the given value or not.
            details (str): What actually happened in the condition.
            category (ConditionCategory): The category to which the condition result belongs.
        """
        self.is_pass = is_pass
        self.details = details
        self.category = category

    def set_name(self, name: str):
        """Set name to be displayed in table.

        Args:
            name (str): Description of the condition to be displayed.
        """
        self.name = name

    def get_sort_value(self):
        """Return sort value of the result."""
        if self.is_pass:
            return 3
        elif self.category == ConditionCategory.FAIL:
            return 1
        else:
            return 2

    def get_icon(self):
        """Return icon of the result to display."""
        if self.is_pass:
            return '<div style="color: green;text-align: center">\U00002713</div>'
        elif self.category == ConditionCategory.FAIL:
            return '<div style="color: red;text-align: center">\U00002716</div>'
        else:
            return '<div style="color: orange;text-align: center;font-weight:bold">\U00000021</div>'

    def __repr__(self):
        """Return string representation for printing."""
        return str(vars(self))


class CheckResult:
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Attributes:
        value (Any): Value calculated by check. Can be used to decide if decidable check passed.
        display (Dict): Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    """

    value: t.Any
    header: str
    display: t.List[t.Union[t.Callable, str, pd.DataFrame, Styler]]
    condition_results: t.List[ConditionResult]
    check: typing.ClassVar

    def __init__(self, value, header: str = None, display: t.Any = None):
        """Init check result.

        Args:
            value (Any): Value calculated by check. Can be used to decide if decidable check passed.
            header (str): Header to be displayed in python notebook.
            check (Class): The check class which created this result. Used to extract the summary to be
            displayed in notebook.
            display (List): Objects to be displayed (dataframe or function or html)
        """
        self.value = value
        self.header = header
        self.condition_results = []

        if display is not None and not isinstance(display, t.List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, t.Callable, Styler)):
                raise DeepchecksValueError(f'Can\'t display item of type: {type(item)}')

    def _ipython_display_(self):
        display_html(f'<h4>{self.get_header()}</h4>', raw=True)
        if hasattr(self.check, '__doc__'):
            docs = self.check.__doc__ or ''
            # Take first non-whitespace line.
            summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')
            display_html(f'<p>{summary}</p>', raw=True)

        for item in self.display:
            if isinstance(item, (pd.DataFrame, Styler)):
                display_dataframe(item)
            elif isinstance(item, str):
                display_html(item, raw=True)
            elif isinstance(item, t.Callable):
                item()
                plt.show()
            else:
                raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            display_html('<p><b>&#x2713;</b> Nothing found</p>', raw=True)

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.value.__repr__()

    def get_header(self):
        """Return header for display. if header was defined return it, else extract name of check class."""
        return self.header or self.check.name()

    def set_condition_results(self, results: t.List[ConditionResult]):
        """Set the conditions results for current check result."""
        self.conditions_results = results

    def have_conditions(self):
        """Return if this check have condition results."""
        return bool(self.conditions_results)

    def have_display(self):
        """Return if this check have dsiplay."""
        return bool(self.display)

    def passed_conditions(self):
        """Return if this check have not passing condition results."""
        return all((r.is_pass for r in self.conditions_results))

    def get_conditions_sort_value(self):
        """Get largest sort value of the conditions results."""
        return max([r.get_sort_value() for r in self.conditions_results])


def wrap_run(func, class_instance):
    """Wrap the run function of checks, and sets the `check` property on the check result."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        result.check = class_instance.__class__
        return result
    return wrapped


class BaseCheck(metaclass=abc.ABCMeta):
    """Base class for check."""

    _conditions: OrderedDict
    _conditions_index: int

    def __init__(self):
        self._conditions = OrderedDict()
        self._conditions_index = 0
        # Replace the run function with wrapped run function
        setattr(self, 'run', wrap_run(getattr(self, 'run'), self))

    def conditions_decision(self, result: CheckResult) -> t.List[ConditionResult]:
        """Run conditions on given result."""
        results = []
        condition: Condition
        for condition in self._conditions.values():
            output = condition.function(result.value, **condition.params)
            if isinstance(output, bool):
                output = ConditionResult(output)
            elif not isinstance(output, ConditionResult):
                raise DeepchecksValueError(f'Invalid return type from condition {condition.name}, got: {type(output)}')
            output.set_name(condition.name)
            results.append(output)
        return results

    def add_condition(self, name: str, condition_func: t.Callable[..., t.Union[ConditionResult, bool]], **params):
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

    def __repr__(self, tabs=0, prefix=''):
        """Representation of check as string.

        Args:
            tabs (int): number of tabs to shift by the output
        """
        tab_chr = '\t'
        params = self.params()
        if params:
            params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
            params_str = f'({params_str})'
        else:
            params_str = ''

        name = prefix + self.__class__.__name__
        check_str = f'{tab_chr * tabs}{name}{params_str}'
        if self._conditions:
            conditions_str = ''.join([f'\n{tab_chr * (tabs + 2)}{i}: {s.name}' for i, s in self._conditions.items()])
            return f'{check_str}\n{tab_chr * (tabs + 1)}Conditions:{conditions_str}'
        else:
            return check_str

    def params(self) -> t.Dict:
        """Return parameters to show when printing the check."""
        return {k: v for k, v in vars(self).items()
                if not k.startswith('_') and v is not None and not callable(v)}

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
            raise DeepchecksValueError(f'Index {index} of conditions does not exists')
        self._conditions.pop(index)

    @classmethod
    def name(cls):
        """Name of class in split camel case."""
        return split_camel_case(cls.__name__)

    # == Checks validation utilities ==

    @classmethod
    def do_datasets_have_label(cls, *datasets: 'dataset.Dataset') -> t.Tuple[pd.Series, ...]:
        """TODO: add comments"""
        check_name = cls.__name__

        if any(d.label_name is None for d in datasets):
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have a label column')

        return tuple(
            t.cast(pd.Series, d.label_col)
            for d in datasets
        )

    @classmethod
    def do_datasets_have_features(cls, *datasets: 'dataset.Dataset') -> t.Tuple[pd.DataFrame, ...]:
        """TODO: add comments"""
        check_name = cls.__name__

        if any(d.features_columns is None for d in datasets):
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have features columns!')

        return tuple(
            t.cast(pd.DataFrame, d.features_columns)
            for d in datasets
        )

    @classmethod
    def do_datasets_have_date(cls, *datasets: 'dataset.Dataset') -> t.Tuple[pd.Series, ...]:
        """TODO: add comments"""
        check_name = cls.__name__

        if any(d.date_name is None for d in datasets):
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have a date column')

        return tuple(
            t.cast(pd.Series, d.date_col)
            for d in datasets
        )

    @classmethod
    def do_datasets_have_index(cls, *datasets: 'dataset.Dataset') -> t.Tuple[pd.Series, ...]:
        """TODO: add comments"""
        check_name = cls.__name__
        
        if any(d.index_name is None for d in datasets):
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have an index column')
        
        return tuple(
            t.cast(pd.Series, d.index_col)
            for d in datasets
        )

    @classmethod
    def do_datasets_share_same_features(cls, *datasets: 'dataset.Dataset') -> t.List[Hashable]:
        """TODO: add comments"""
        check_name = cls.__name__
        if not dataset.Dataset.share_same_features(*datasets):
            raise DeepchecksValueError(f'Check {check_name} requires datasets to share the same features')
        return datasets[0].features

    @classmethod
    def do_datasets_share_same_categorical_features(cls, *datasets: 'dataset.Dataset') -> t.List[Hashable]:
        """TODO: add comments"""
        check_name = cls.__name__
        if not dataset.Dataset.share_same_categorical_features(*datasets):
            raise DeepchecksValueError(
                f'Check {check_name} requires datasets to share '
                'the same categorical features. Possible reason is that some columns were'
                'inferred incorrectly as categorical features. To fix this, manually edit the '
                'categorical features using Dataset(cat_features=<list_of_features>'
            )
        return datasets[0].cat_features

    @classmethod
    def do_datasets_share_same_label(cls, *datasets: 'dataset.Dataset') -> Hashable:
        """TODO: add comments"""
        check_name = cls.__name__
        if not dataset.Dataset.share_same_label(*datasets):
            raise DeepchecksValueError(f'Check {check_name} requires datasets to share the same label')
        return t.cast(Hashable, datasets[0].label_name)

    @classmethod
    def are_not_empty_datasets(cls, *values: object) -> t.Tuple['dataset.Dataset', ...]:
        """TODO:"""
        if len(values) == 0:
            return tuple()
        
        invalid_value = next(
            (it for it in values if not isinstance(it, dataset.Dataset)),
            None
        )

        if invalid_value is not None:
            raise DeepchecksValueError(
                'Check requires input value(s) to be of type Dataset. '
                f'Instead got: {type(invalid_value).__name__}'
            )

        empty_dataset = next(
            (it for it in values if len(t.cast(dataset.Dataset, it).data) == 0),
            None
        )
        
        if empty_dataset is not None:
            raise DeepchecksValueError('Check requires non-empty datasets!')

        return t.cast(t.Tuple[dataset.Dataset, ...], values)

    @classmethod
    def ensure_not_empty_dataset(cls, value: object) -> 'dataset.Dataset':
        """TODO: add comments"""
        check_name = cls.__name__
        return ensure_not_empty_dataset(value, error_messages={
            'empty': f'Check {check_name} required a non-empty dataset',
            'incorrect_value': (
                f'Check {check_name} requires dataset to be of type Dataset or Dataframe.'
                f'Instead got: {type(value).__name__}'
            )
        })

    @classmethod
    def infer_task_type(
        cls,
        model: t.Any,
        dataset: 'dataset.Dataset',
        expected_types: t.Sequence[ModelType],
    ) -> ModelType:
        """TODO: add coments"""
        check_name = cls.__name__
        task_type = infer_task_type(model, dataset)

        if task_type not in expected_types:
            stringified_types = ','.join(e.value for e in expected_types)
            raise DeepchecksValueError(
                f'Check {check_name} expected model to be a type from {stringified_types}, '
                f'but received model of type: {task_type.value}'
            )

        return task_type


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


class TrainTestBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    @abc.abstractmethod
    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    @abc.abstractmethod
    def run(self, model) -> CheckResult:
        """Define run signature."""
        pass


@dataclass
class CheckFailure:
    """Class which holds a run exception of a check."""

    check: t.Any  # Check class, not instance
    exception: Exception
