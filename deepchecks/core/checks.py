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
import json
from collections import OrderedDict
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union, cast

from typing_extensions import NotRequired, Self, TypedDict

from deepchecks import __version__
from deepchecks.core import check_result as check_types  # pylint: disable=unused-import
from deepchecks.core.condition import Condition, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.function import initvars
from deepchecks.utils.strings import get_docs_summary, split_camel_case

from . import common

__all__ = [
    'DatasetKind',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck'
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
    module_name: str
    class_name: str
    version: NotRequired[str]
    params: Dict[Any, Any]


class BaseCheck(abc.ABC):
    """Base class for check."""

    _conditions: OrderedDict
    _conditions_index: int

    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        self._conditions = OrderedDict()
        self._conditions_index = 0
        self.n_samples = kwargs.get('n_samples')  # None indicates that the check will run on the entire dataset

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
        return initvars(self, include_defaults=show_defaults)

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

    def to_json(self, indent: int = 3, include_version: bool = True, include_defaults: bool = True) -> str:
        """Serialize check instance to JSON string."""
        conf = self.config(include_version=include_version, include_defaults=include_defaults)
        return json.dumps(conf, indent=indent)

    @classmethod
    def from_json(
            cls: Type[Self],
            conf: str,
            version_unmatch: 'common.VersionUnmatchAction' = 'warn'
    ) -> Self:
        """Deserialize check instance from JSON string."""
        check_conf = json.loads(conf)
        return cls.from_config(check_conf, version_unmatch=version_unmatch)

    def _prepare_config(
            self,
            params: Dict[str, Any],
            include_version: bool = True
    ) -> CheckConfig:
        module_name, type_name = common.importable_name(self)
        conf = CheckConfig(
            class_name=type_name,
            module_name=module_name,
            params=params,
        )
        if include_version is True:
            conf['version'] = __version__
        return conf

    def config(self, include_version: bool = True, include_defaults: bool = True) -> CheckConfig:
        """Return check configuration (conditions' configuration not yet supported).

        Returns
        -------
        CheckConfig
            includes the checks class name, params, and module name.
        """
        # NOTE:
        # default implementation of the config method makes an assumption
        # that Check type '__init__' method represents check instance internal
        # state, it is true for some of our checks but not for all.
        # To not do any implicit magical stuff the simplest solution for those
        # check types for which this assumption is not true is to override the
        # 'config' method.
        # Another important assumption about the config method and it return value is
        # that a check instance state (config.params) is represented by simple builtin
        # types that can be serialized to json/yaml and will not lose type information
        # between serialization/deserialization that might cause check instance to fall
        # after it recreation from config.
        # Again if that is not true for some sub-check it must override this method and to ensure
        # this assumption
        return self._prepare_config(
            params=initvars(self, include_defaults=include_defaults),
            include_version=include_version
        )

    @classmethod
    def from_config(
            cls: Type[Self],
            conf: CheckConfig,
            version_unmatch: 'common.VersionUnmatchAction' = 'warn'
    ) -> Self:
        """Return check object from a CheckConfig object.

        Parameters
        ----------
        conf : Dict[Any, Any]

        Returns
        -------
        BaseCheck
            the check class object from given config
        """
        # NOTE:
        # within the method we need to treat conf as a dict with unknown structure/content
        check_conf = cast(Dict[str, Any], conf)
        check_conf = common.validate_config(check_conf, version_unmatch=version_unmatch)
        type_ = common.import_type(
            type_name=check_conf['class_name'],
            module_name=check_conf['module_name'],
            base=cls
        )
        return type_(**check_conf['params'])

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
    def run(self, dataset, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()


class TrainTestBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type: ClassVar[Optional[Type[Any]]] = None  # TODO: Base context type

    @abc.abstractmethod
    def run(self, train_dataset, test_dataset, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type: ClassVar[Optional[Type[Any]]] = None  # TODO: Base context type

    @abc.abstractmethod
    def run(self, model, **kwargs) -> 'check_types.CheckResult':
        """Run check."""
        raise NotImplementedError()
