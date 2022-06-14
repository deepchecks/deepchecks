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
from typing import Any, List, Mapping, Union

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import (BaseCheck, DatasetKind, ModelOnlyBaseCheck, SingleDatasetBaseCheck,
                                    TrainTestBaseCheck)
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


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    def run(self, dataset, model=None, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train=dataset,
            model=model,
            **kwargs
        )
        result = self.run_logic(context, dataset_kind=DatasetKind.TRAIN)
        context.finalize_check_result(result, self, DatasetKind.TRAIN)
        return result

    @abc.abstractmethod
    def run_logic(self, context, dataset_kind) -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class TrainTestCheck(TrainTestBaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type = Context

    def run(self, train_dataset, test_dataset, model=None, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train_dataset,
            test_dataset,
            model=model,
            **kwargs
        )
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class ModelOnlyCheck(ModelOnlyBaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type = Context

    def run(self, model, **kwargs) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(model=model, **kwargs)  # pylint: disable=not-callable
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))


class ModelComparisonCheck(BaseCheck):
    """Parent class for check that compares between two or more models."""

    def run(self,
            train_datasets: Union[Dataset, List[Dataset]],
            test_datasets: Union[Dataset, List[Dataset]],
            models: Union[List[Any], Mapping[str, Any]]
            ) -> CheckResult:
        """Initialize context and pass to check logic."""
        context = ModelComparisonContext(train_datasets, test_datasets, models)
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, multi_context: ModelComparisonContext) -> CheckResult:
        """Implement here logic of check."""
        raise NotImplementedError()
