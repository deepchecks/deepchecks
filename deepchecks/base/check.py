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
# pylint: disable=broad-except
import abc
import enum
import inspect
import re
import traceback
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, List, Union, Dict, cast, Mapping
import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure

from deepchecks.base.dataset import Dataset
from deepchecks.base.display_pandas import display_conditions_table, display_dataframe
from deepchecks.utils.strings import split_camel_case
from deepchecks.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.utils.ipython import is_ipython_display
from deepchecks.utils.metrics import task_type_check
from deepchecks.utils.validation import validate_model


__all__ = [
    'CheckResult',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ModelComparisonBaseCheck',
    'ConditionResult',
    'ConditionCategory',
    'CheckFailure',
    'ModelComparisonContext'
]


class Condition:
    """Contain condition attributes."""

    name: str
    function: Callable
    params: Dict

    def __init__(self, name: str, function: Callable, params: Dict):
        if not isinstance(function, Callable):
            raise DeepchecksValueError(f'Condition must be a function `(Any) -> Union[ConditionResult, bool]`, '
                                       f'but got: {type(function).__name__}')
        if not isinstance(name, str):
            raise DeepchecksValueError(f'Condition name must be of type str but got: {type(name).__name__}')
        self.name = name
        self.function = function
        self.params = params

    def __call__(self, *args, **kwargs) -> 'ConditionResult':
        result = cast(ConditionResult, self.function(*args, **kwargs))
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

    @property
    def priority(self) -> int:
        """Return priority of the current condition.

        This value is primarily used to determine the order in which
        conditions should be displayed.

        Returns:
            int: condition priority value;
        """
        if self.is_pass is True:
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

    value: Any
    header: str
    display: List[Union[Callable, str, pd.DataFrame, Styler]]
    conditions_results: List[ConditionResult]
    check: 'BaseCheck'

    def __init__(self, value, header: str = None, display: Any = None):
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
        self.conditions_results = []

        if display is not None and not isinstance(display, List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Styler, Callable, BaseFigure)):
                raise DeepchecksValueError(f'Can\'t display item of type: {type(item)}')

    def _ipython_display_(self, show_conditions=True, unique_id=None):
        if unique_id:
            check_id = f'{self.check.__class__.__name__}_{unique_id}'
            display_html(f'<h4 id="{check_id}">{self.get_header()}</h4>', raw=True)
        else:
            display_html(f'<h4>{self.get_header()}</h4>', raw=True)
        if hasattr(self.check.__class__, '__doc__'):
            docs = self.check.__class__.__doc__ or ''
            # Take first non-whitespace line.
            summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')
            display_html(f'<p>{summary}</p>', raw=True)
        if self.conditions_results and show_conditions:
            display_html('<h5>Conditions Summary</h5>', raw=True)
            display_conditions_table(self, unique_id)
            display_html('<h5>Additional Outputs</h5>', raw=True)
        for item in self.display:
            if isinstance(item, (pd.DataFrame, Styler)):
                display_dataframe(item)
            elif isinstance(item, str):
                display_html(item, raw=True)
            elif isinstance(item, BaseFigure):
                item.show()
            elif callable(item):
                try:
                    item()
                    plt.show()
                except Exception as exc:
                    display_html(f'Error in display {str(exc)}', raw=True)
            else:
                raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            display_html('<p><b>&#x2713;</b> Nothing found</p>', raw=True)
        if unique_id:
            display_html(f'<br><a href="#summary_{unique_id}" style="font-size: 14px">Go to top</a>', raw=True)

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return f'{self.get_header()}: {self.value}'

    def get_header(self):
        """Return header for display. if header was defined return it, else extract name of check class."""
        return self.header or self.check.name()

    def process_conditions(self):
        """Process the conditions results from current result and check."""
        self.conditions_results = self.check.conditions_decision(self)

    def have_conditions(self) -> bool:
        """Return if this check have condition results."""
        return bool(self.conditions_results)

    def have_display(self) -> bool:
        """Return if this check have dsiplay."""
        return bool(self.display)

    def passed_conditions(self):
        """Return if this check have not passing condition results."""
        return all((r.is_pass for r in self.conditions_results))

    @property
    def priority(self) -> int:
        """Return priority of the current result.

        This value is primarly used to determine suite output order.
        The logic is next:
            - if at least one condition did not pass and is of category 'FAIL', return 1;
            - if at least one condition did not pass and is of category 'WARN', return 2;
            - if check result do not have assigned conditions, return 3
            - if all conditions passed, return 4 ;

        Returns:
            int: priority of the cehck result.
        """
        if not self.have_conditions:
            return 3

        for c in self.conditions_results:
            if c.is_pass is False and c.category == ConditionCategory.FAIL:
                return 1
            if c.is_pass is False and c.category == ConditionCategory.WARN:
                return 2

        return 4

    def show(self, show_conditions=True, unique_id=None):
        """Display check result."""
        if is_ipython_display():
            self._ipython_display_(show_conditions=show_conditions, unique_id=unique_id)
        else:
            print(self)


def wrap_run(func, class_instance):
    """Wrap the run function of checks, and sets the `check` property on the check result."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, CheckResult):
            raise DeepchecksValueError(f'Check {class_instance.name()} expected to return CheckResult but got: '
                                       + type(result).__name__)
        result.check = class_instance
        result.process_conditions()
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

    def conditions_decision(self, result: CheckResult) -> List[ConditionResult]:
        """Run conditions on given result."""
        results = []
        condition: Condition
        for condition in self._conditions.values():
            try:
                output = condition.function(result.value, **condition.params)
            except Exception as e:
                msg = f'Exception in condition: {e.__class__.__name__}: {str(e)}'
                output = ConditionResult(False, msg, ConditionCategory.WARN)
            if isinstance(output, bool):
                output = ConditionResult(output)
            elif not isinstance(output, ConditionResult):
                raise DeepchecksValueError(f'Invalid return type from condition {condition.name}, got: {type(output)}')
            output.set_name(condition.name)
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

    def params(self) -> Dict:
        """Return parameters to show when printing the check."""
        init_params = inspect.signature(self.__init__).parameters

        return {k: v for k, v in vars(self).items()
                if k in init_params and v != init_params[k].default}

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


class SingleDatasetBaseCheck(BaseCheck):
    """Parent class for checks that only use one dataset."""

    @abc.abstractmethod
    def run(self, dataset, model=None) -> CheckResult:
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


class ModelComparisonContext:
    """Contain processed input for model comparison checks."""

    def __init__(self,
                 train_datasets: Union[Dataset, List[Dataset]],
                 test_datasets: Union[Dataset, List[Dataset]],
                 models: Union[List[Any], Mapping[str, Any]]
                 ):
        """Preprocess the parameters."""
        # Validations
        if isinstance(train_datasets, Dataset) and isinstance(test_datasets, List):
            raise DeepchecksNotSupportedError('Single train dataset with multiple test datasets is not supported.')

        if not isinstance(models, (List, Mapping)):
            raise DeepchecksValueError('`models` must be a list or dictionary for compare models checks.')
        if len(models) < 2:
            raise DeepchecksValueError('`models` must receive 2 or more models')
        # Some logic to assign names to models
        if isinstance(models, List):
            models_dict = {}
            for m in models:
                model_type = type(m).__name__
                numerator = 1
                name = model_type
                while name in models_dict:
                    name = f'{model_type}_{numerator}'
                    numerator += 1
                models_dict[name] = m
            models = models_dict

        if not isinstance(train_datasets, List):
            train_datasets = [train_datasets] * len(models)
        if not isinstance(test_datasets, List):
            test_datasets = [test_datasets] * len(models)

        if len(train_datasets) != len(models):
            raise DeepchecksValueError('number of train_datasets must equal to number of models')
        if len(test_datasets) != len(models):
            raise DeepchecksValueError('number of test_datasets must equal to number of models')

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.model_names = list(models.keys())
        self.models = list(models.values())

        # Additional validations
        self.task_type = None
        for i in range(len(models)):
            train = self.train_datasets[i]
            test = self.test_datasets[i]
            model = self.models[i]
            Dataset.validate_dataset(train)
            Dataset.validate_dataset(test)
            train.validate_label()
            train.validate_features()
            train.validate_shared_features(test)
            train.validate_shared_label(test)
            validate_model(train, model)
            curr_task_type = task_type_check(model, train)
            if self.task_type is None:
                self.task_type = curr_task_type
            elif curr_task_type != self.task_type:
                raise DeepchecksNotSupportedError('Got models of different task types')

    def __len__(self):
        """Return number of models."""
        return len(self.models)

    def __iter__(self):
        """Return iterator over context objects."""
        return zip(self.train_datasets, self.test_datasets, self.models, self.model_names)


class ModelComparisonBaseCheck(BaseCheck):
    """Parent class for check that compares between two or more models."""

    def run(self,
            train_datasets: Union[Dataset, List[Dataset]],
            test_datasets: Union[Dataset, List[Dataset]],
            models: Union[List[Any], Mapping[str, Any]]
            ) -> CheckResult:
        """Initialize context and pass to check logic."""
        return self.run_logic(ModelComparisonContext(train_datasets, test_datasets, models))

    @abc.abstractmethod
    def run_logic(self, context: ModelComparisonContext):
        """Implement here logic of check."""
        pass


class CheckFailure:
    """Class which holds a run exception of a check."""

    def __init__(self, check: BaseCheck, exception: Exception):
        self.check = check
        self.exception = exception
        self.header = check.name()

    def __repr__(self):
        """Return string representation."""
        tb_str = traceback.format_exception(etype=type(self.exception), value=self.exception,
                                            tb=self.exception.__traceback__)
        return ''.join(tb_str)
