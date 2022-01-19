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
import base64
import inspect
import io
import traceback
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, List, Sequence, Union, Dict, Mapping, cast

import jsonpickle
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.core.display import display_html
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
import plotly

from deepchecks.base.condition import Condition, ConditionCategory, ConditionResult
from deepchecks.base.dataset import Dataset
from deepchecks.base.display_pandas import dataframe_to_html, get_conditions_table
from deepchecks.utils.typing import Hashable, BasicModel
from deepchecks.utils.strings import get_docs_summary, split_camel_case
from deepchecks.utils.ipython import is_ipython_display
from deepchecks.utils.metrics import task_type_check, ModelType
from deepchecks.utils.validation import validate_model
from deepchecks.errors import (
    DeepchecksValueError,
    DeepchecksNotSupportedError,
    DatasetValidationError,
    ModelValidationError
)


__all__ = [
    'CheckResult',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'CheckFailure',
]


def _save_all_open_figures():
    figs = [plt.figure(n) for n in plt.get_fignums()]
    images = []
    for fig in figs:
        bio = io.BytesIO()
        fig.savefig(bio, format='png')
        encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
        images.append(encoded)
        fig.clear()
    return images


_CONDITIONS_HEADER = '<h5>Conditions Summary</h5>'
_ADDITIONAL_OUTPUTS_HEADER = '<h5>Additional Outputs</h5>'


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

    def display_check(self, unique_id: str = None, as_widget: bool = False,
                      show_additional_outputs=True):  # pragma: no cover
        """Display the check result or return the display as widget.

        Args:
            unique_id (str):
                The unique id given by the suite that displays the check.
            as_widget (bool):
                Boolean that controls if to display the check regulary or if to return a widget.
            show_additional_outputs (bool):
                Boolean that controls if to show additional outputs.
        Returns:
            Widget representation of the display if as_widget is True.
        """
        if as_widget:
            box = widgets.VBox()
            box_children = []
        check_html = ''
        if unique_id:
            check_id = f'{self.check.__class__.__name__}_{unique_id}'
            check_html += f'<h4 id="{check_id}">{self.get_header()}</h4>'
        else:
            check_html += f'<h4>{self.get_header()}</h4>'
        if hasattr(self.check.__class__, '__doc__'):
            summary = get_docs_summary(self.check)
            check_html += f'<p>{summary}</p>'
        if self.conditions_results:
            check_html += _CONDITIONS_HEADER
            check_html += dataframe_to_html(get_conditions_table(self, unique_id))
        if show_additional_outputs:
            check_html += _ADDITIONAL_OUTPUTS_HEADER
            for item in self.display:
                if isinstance(item, (pd.DataFrame, Styler)):
                    check_html += dataframe_to_html(item)
                elif isinstance(item, str):
                    check_html += f'<div>{item}</div>'
                elif isinstance(item, BaseFigure):
                    if as_widget:
                        box_children.append(widgets.HTML(check_html))
                        box_children.append(go.FigureWidget(data=item))
                    else:
                        display_html(check_html, raw=True)
                        item.show()
                    check_html = ''
                elif callable(item):
                    try:
                        if as_widget:
                            plt_out = widgets.Output()
                            with plt_out:
                                item()
                                plt.show()
                            box_children.append(widgets.HTML(check_html))
                            box_children.append(plt_out)
                        else:
                            display_html(check_html, raw=True)
                            item()
                            plt.show()
                        check_html = ''
                    except Exception as exc:
                        check_html += f'Error in display {str(exc)}'
                else:
                    raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            check_html += '<p><b>&#x2713;</b> Nothing found</p>'
        if unique_id:
            check_html += f'<br><a href="#summary_{unique_id}" style="font-size: 14px">Go to top</a>'
        if as_widget:
            box_children.append(widgets.HTML(check_html))
            box.children = box_children
            return box
        display_html(check_html, raw=True)

    def _display_to_json(self):
        displays = []
        old_backend = matplotlib.get_backend()
        for item in self.display:
            if isinstance(item, Styler):
                displays.append(('dataframe', item.data.to_json(orient='records')))
            elif isinstance(item, pd.DataFrame):
                displays.append(('dataframe', item.to_json(orient='records')))
            elif isinstance(item, str):
                displays.append(('html', item))
            elif isinstance(item, BaseFigure):
                displays.append(('plotly', item.to_json()))
            elif callable(item):
                try:
                    matplotlib.use('Agg')
                    item()
                    displays.append(('plt', _save_all_open_figures()))
                except Exception:
                    displays.append(('plt', ''))
            else:
                matplotlib.use(old_backend)
                raise Exception(f'Unable to create json for item of type: {type(item)}')
        matplotlib.use(old_backend)
        return displays

    def to_json(self, with_display: bool = True):
        """Return check result as json.

        Args:
            with_display (bool): controls if to serialize display or not

        Returns:
            json in the format:
            {'name': .., 'params': .., 'header': ..,
             'summary': .., 'conditions_table': .., 'value', 'display': ..}
        """
        check_name = self.check.name()
        parameters = self.check.params()
        header = self.get_header()
        result_json = {'name': check_name, 'params': parameters, 'header': header,
                       'summary': get_docs_summary(self.check)}
        if self.conditions_results:
            cond_df = get_conditions_table(self)
            result_json['conditions_table'] = cond_df.data.to_json(orient='records')
        if isinstance(self.value, pd.DataFrame):
            result_json['value'] = self.value.to_json()
        elif isinstance(self.value, np.ndarray):
            result_json['value'] = self.value.tolist()
        else:
            result_json['value'] = self.value
        if with_display:
            display_json = self._display_to_json()
            result_json['display'] = display_json
        return jsonpickle.dumps(result_json)

    @staticmethod
    def display_from_json(json_data):
        """Display the check result from a json received from a to_json."""
        json_data = jsonpickle.loads(json_data)
        if json_data.get('display') is None:
            return
        header = json_data['header']
        summary = json_data['summary']
        display_html(f'<h4>{header}</h4>', raw=True)
        display_html(f'<p>{summary}</p>', raw=True)
        if json_data.get('conditions_table'):
            display_html(_CONDITIONS_HEADER, raw=True)
            conditions_table = pd.read_json(json_data['conditions_table'], orient='records')
            display_html(dataframe_to_html(conditions_table.style.hide_index()), raw=True)
        display_html(_ADDITIONAL_OUTPUTS_HEADER, raw=True)
        for display_type, value in json_data['display']:
            if display_type == 'html':
                display_html(value, raw=True)
            elif display_type in ['conditions', 'dataframe']:
                df: pd.DataFrame = pd.read_json(value, orient='records')
                display_html(dataframe_to_html(df), raw=True)
            elif display_type == 'plotly':
                plotly_json = io.StringIO(value)
                plotly.io.read_json(plotly_json).show()
            elif display_type == 'plt':
                display_html(f'<img src=\'data:image/png;base64,{value}\'>', raw=True)
            else:
                raise ValueError(f'Unexpected type of display received: {display_type}')

    def _ipython_display_(self, unique_id=None, as_widget=False,
                          show_additional_outputs=True):
        check_widget = self.display_check(unique_id=unique_id, as_widget=as_widget,
                                          show_additional_outputs=show_additional_outputs,)
        if as_widget:
            display_html(check_widget)

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
            - if all conditions passed, return 4;

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

    def show(self, unique_id=None, show_additional_outputs=True):
        """Display check result."""
        if is_ipython_display():
            self._ipython_display_(unique_id=unique_id,
                                   show_additional_outputs=show_additional_outputs)
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

    # NOTE: next set of private functions exists to unify error messages across all checks

    @classmethod
    def _datasets_share_features(cls, datasets: List[Dataset]) -> List[Hashable]:
        """Verify that all provided datasets share same features, otherwise raise an exception.

        Args:
            datasets (List[Dataset]): list of datasets to validate

        Returns:
            List[Hashable]: list of features

        Raises:
            DatasetValidationError: if datasets do not share same features;
        """
        if Dataset.datasets_share_features(datasets) is False:
            raise DatasetValidationError('Check requires Datasets to share the same features')
        return datasets[0].features

    @staticmethod
    def _datasets_share_categorical_features(datasets: List['Dataset']) -> List[Hashable]:
        """Verify that all provided datasets share same categorical features, otherwise raise an exception.

        Args:
            datasets (List[Dataset]): list of datasets to validate

        Returns:
            List[Hashable]: list of categorical features

        Raises:
            DatasetValidationError: if datasets do not share same categorical features;
        """
        if Dataset.datasets_share_categorical_features(datasets) is False:
            raise DatasetValidationError(
                'Check requires datasets to share '
                'the same categorical features. Possible reason is that some columns were'
                'inferred incorrectly as categorical features. To fix this, manually edit the '
                'categorical features using Dataset(cat_features=<list_of_features>'
            )
        return datasets[0].cat_features

    @staticmethod
    def _datasets_share_label(datasets: List['Dataset']) -> Hashable:
        """Verify that all provided datasets share same label, otherwise raise an exception.

        Args:
            datasets (List[Dataset]): list of datasets to validate

        Returns:
            Hashable: name of the label column

        Raises:
            DatasetValidationError: if datasets do not share same label;
        """
        if Dataset.datasets_share_label(datasets) is False:
            raise DatasetValidationError('Check requires Datasets to have and to share the same label')
        return cast(Hashable, datasets[0].label_name)

    @staticmethod
    def _dataset_has_label(dataset: Dataset) -> pd.Series:
        """Verify that provided dataset has label, otherwise raise an exception.

        Args:
            datasets (Dataset): dataset to validate

        Returns:
            pandas.Series: label column

        Raises:
            DatasetValidationError: if dataset does not have label;
        """
        if dataset.label_col is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without label')
        return dataset.label_col

    @staticmethod
    def _dataset_has_features(dataset: Dataset) -> pd.DataFrame:
        """Verify that provided dataset has features, otherwise raise an exception.

        Args:
            datasets (Dataset): dataset to validate

        Returns:
            pandas.DataFrame: features dataframe

        Raises:
            DatasetValidationError: if dataset does not have features;
        """
        if (
            dataset.features_columns is None
            or len(dataset.features_columns.columns) == 0
        ):
            raise DatasetValidationError('Check is irrelevant for Datasets without features')
        return dataset.features_columns

    @staticmethod
    def _dataset_has_date(dataset: Dataset) -> pd.Series:
        """Verify that provided dataset has date column, otherwise raise an exception.

        Args:
            datasets (Dataset): dataset to validate

        Returns:
            pandas.Series: date column

        Raises:
            DatasetValidationError: if dataset does not have date column;
        """
        if dataset.datetime_col is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without datetime column')
        return dataset.datetime_col

    @staticmethod
    def _dataset_has_index(dataset: Dataset) -> pd.Series:
        """Verify that provided dataset has index, otherwise raise an exception.

        Args:
            datasets (Dataset): dataset to validate

        Returns:
            pandas.Series: dataset index column

        Raises:
            DatasetValidationError: if dataset does not have index;
        """
        if dataset.index_col is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without an index')
        return dataset.index_col

    @staticmethod
    def _verify_model_type(
        model: BasicModel,
        dataset: 'Dataset',
        expected_types: Sequence[ModelType]
    ) -> ModelType:
        """Verify that provided model is of an expected type, otherwise raise an exception.

        Returns:
            ModelType: type of the provided model

        Raises:
            ModelValidationError: if unexpected model type is provided;
        """
        task_type = task_type_check(model, dataset)
        if task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )
        return task_type


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


class CheckFailure:
    """Class which holds a run exception of a check."""

    def __init__(self, check: BaseCheck, exception: Exception, header_suffix: str = ''):
        self.check = check
        self.exception = exception
        self.header = check.name() + header_suffix

    def to_json(self, with_display: bool = True):
        """Return check failure as json.

        Args:
            with_display (bool): controls if to serialize display or not

        Returns:
            {'name': .., 'params': .., 'header': .., 'display': ..}
        """
        check_name = self.check.name()
        parameters = self.check.params()
        result_json = {'name': check_name, 'params': parameters, 'header': self.header}
        if with_display:
            result_json['display'] = [('str', str(self.exception))]
        return jsonpickle.dumps(result_json)

    def __repr__(self):
        """Return string representation."""
        tb_str = traceback.format_exception(etype=type(self.exception), value=self.exception,
                                            tb=self.exception.__traceback__)
        return ''.join(tb_str)


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
            train = Dataset.ensure_not_empty_dataset(train)
            test = Dataset.ensure_not_empty_dataset(test)
            BaseCheck._dataset_has_label(train)
            BaseCheck._dataset_has_features(train)
            BaseCheck._datasets_share_features([train, test])
            BaseCheck._datasets_share_label([train, test])
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
    def run_logic(self, context: ModelComparisonContext) -> CheckResult:
        """Implement here logic of check."""
        pass
