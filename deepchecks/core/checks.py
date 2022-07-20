# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the base checks."""
# pylint: disable=broad-except
import abc
import enum
import importlib
from collections import OrderedDict
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

from typing_extensions import TypedDict

from deepchecks.core import check_result as check_types  # pylint: disable=unused-import
from deepchecks.core.condition import Condition, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.function import initvars
from deepchecks.utils.strings import get_docs_summary, split_camel_case

__all__ = [
    'DatasetKind',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ReduceMixin'
]


class DatasetKind(enum.Enum):
    """Represents in single dataset checks, which dataset is currently worked on."""

    TRAIN = 'Train'
    TEST = 'Test'


class CheckMetadata(TypedDict):
    name: str
    params: Dict[Any, Any]
    summary: str


class CheckConfig(TypedDict):
    class_name: str
    module_name: Optional[str]
    params: Dict[Any, Any]


class ReduceMixin(abc.ABC):
    """Mixin for reduce_output function."""

    def reduce_output(self, check_result: 'check_types.CheckResult') -> Dict[str, float]:
        """Return the check result as a reduced dict. Being Used for monitoring.

        Parameters
        ----------
        check_result : CheckResult
            The check result.

        Returns
        -------
        Dict[str, float]
            reduced dictionary in format {str: float} (i.e {'AUC': 0.1}), based on the check's original returned value
        """
        raise NotImplementedError('Must implement reduce_output function')


class BaseCheck(abc.ABC):
    """Base class for check."""

    _conditions: OrderedDict
    _conditions_index: int

    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        self._conditions = OrderedDict()
        self._conditions_index = 0

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> 'check_types.CheckResult':
        """Run Check."""
        raise NotImplementedError()

    def conditions_decision(self, result: 'check_types.CheckResult') -> List[ConditionResult]:
        """Run conditions on given result."""
        results = []
        condition: Condition
        for condition in self._conditions.values():
            try:
                output = condition.function(result.value, **condition.params)
            except Exception as e:
                msg = f'Exception in condition: {e.__class__.__name__}: {str(e)}'
                output = ConditionResult(ConditionCategory.ERROR, msg)
            if isinstance(output, bool):
                output = ConditionResult(ConditionCategory.PASS if output else ConditionCategory.FAIL)
            elif not isinstance(output, ConditionResult):
                raise DeepchecksValueError(f'Invalid return type from condition {condition.name}, got: {type(output)}')
            output.set_name(condition.name)
            results.append(output)
        return results

    def add_condition(self, name: str, condition_func: Callable[[Any], Union[ConditionResult, bool]], **params):
        """Add new condition function to the check.

        Parameters
        ----------
        name : str
            Name of the condition. should explain the condition action and parameters
        condition_func : Callable[[Any], Union[List[ConditionResult], bool]]
            Function which gets the value of the check and returns object of List[ConditionResult] or boolean.
        params : dict
            Additional parameters to pass when calling the condition function.

        """
        cond = Condition(name, condition_func, params)
        self._conditions[self._conditions_index] = cond
        self._conditions_index += 1
        return self

    def clean_conditions(self):
        """Remove all conditions from this check instance."""
        self._conditions.clear()
        self._conditions_index = 0

    def remove_condition(self, index: int):
        """Remove given condition by index.

        Parameters
        ----------
        index : int
            index of condtion to remove

        """
        if index not in self._conditions:
            raise DeepchecksValueError(f'Index {index} of conditions does not exists')
        self._conditions.pop(index)

    def params(self, show_defaults: bool = False) -> Dict:
        """Return parameters to show when printing the check."""
        return initvars(self, show_defaults)

    @classmethod
    def name(cls) -> str:
        """Name of class in split camel case."""
        return split_camel_case(cls.__name__)

    def metadata(self, with_doc_link: bool = False) -> CheckMetadata:
        """Return check metadata.

        Parameters
        ----------
        with_doc_link : bool, default False
            whethere to include doc link in summary or not

        Returns
        -------
        Dict[str, Any]
        """
        return CheckMetadata(
            name=self.name(),
            params=self.params(show_defaults=True),
            summary=get_docs_summary(self, with_doc_link)
        )

    def config(self) -> CheckConfig:
        """Return check configuration (conditions' configuration not yet supported).

        Returns
        -------
        CheckConfig
            includes the checks class name, params, and module name.
        """
        conf = CheckConfig(
            class_name=self.__class__.__name__,
            params=self.params(show_defaults=True),
            module_name=self.__module__
        )
        return conf

    @staticmethod
    def from_config(conf: CheckConfig) -> 'BaseCheck':
        """Return check object from a CheckConfig object.

        Parameters
        ----------
        conf : CheckConfig
            the CheckConfig object

        Returns
        -------
        BaseCheck
            the check class object from given config
        """
        module = importlib.import_module(conf['module_name'])
        return getattr(module, conf['class_name'])(**conf['params'])

    def __repr__(self, tabs=0, prefix=''):
        """Representation of check as string.

        Parameters
        ----------
        tabs : int , default: 0
            number of tabs to shift by the output
        prefix

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


class SingleDatasetBaseCheck(BaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type: ClassVar[Optional[Type[Any]]] = None  # TODO: Base context type

    @abc.abstractmethod
    def run(self, dataset, model=None, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()


class TrainTestBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type: ClassVar[Optional[Type[Any]]] = None  # TODO: Base context type

    @abc.abstractmethod
    def run(self, train_dataset, test_dataset, model=None, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type: ClassVar[Optional[Type[Any]]] = None  # TODO: Base context type

    @abc.abstractmethod
    def run(self, model, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()
