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
"""Module for tabular base checks."""
import abc
from functools import wraps
from typing import Any, List, Mapping, Union

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import BaseCheck, ModelOnlyBaseCheck, SingleDatasetBaseCheck, TrainTestBaseCheck
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.tabular import deprecation_warnings  # pylint: disable=unused-import # noqa: F401
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.model_base import ModelComparisonContext

__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'ModelComparisonCheck'
]


def wrap_run(func, check_instance: BaseCheck):
    """Wrap the run function of checks, and sets the `check` property on the check result."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        return check_instance.finalize_check_result(result)

    return wrapped


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, dataset, model=None, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(  # pylint: disable=not-callable
            dataset,
            model=model,
            **kwargs
        ))

    @abc.abstractmethod
    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class TrainTestCheck(TrainTestBaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type = Context

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, train_dataset, test_dataset, model=None, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(  # pylint: disable=not-callable
            train_dataset,
            test_dataset,
            model=model,
            **kwargs
        ))

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class ModelOnlyCheck(ModelOnlyBaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type = Context

    def __init__(self, **kwargs):
        """Initialize the class."""
        super().__init__(**kwargs)
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, model, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(model=model, **kwargs))  # pylint: disable=not-callable

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))


class ModelComparisonCheck(BaseCheck):
    """Parent class for check that compares between two or more models."""

    def __init__(self, **kwargs):
        """Initialize the class."""
        super().__init__(**kwargs)
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self,
            train_datasets: Union[Dataset, List[Dataset]],
            test_datasets: Union[Dataset, List[Dataset]],
            models: Union[List[Any], Mapping[str, Any]]
            ) -> CheckResult:
        """Initialize context and pass to check logic."""
        return self.run_logic(ModelComparisonContext(train_datasets, test_datasets, models))

    @abc.abstractmethod
    def run_logic(self, multi_context: ModelComparisonContext) -> CheckResult:
        """Implement here logic of check."""
        raise NotImplementedError()
