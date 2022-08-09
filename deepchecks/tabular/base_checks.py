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
from typing import List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import (BaseCheck, DatasetKind, ModelOnlyBaseCheck, SingleDatasetBaseCheck,
                                    TrainTestBaseCheck)
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.tabular import deprecation_warnings  # pylint: disable=unused-import # noqa: F401
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.model_base import ModelComparisonContext
from deepchecks.utils.decorators import deprecate_kwarg
from deepchecks.utils.typing import BasicModel

__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'ModelComparisonCheck'
]


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    @deprecate_kwarg(old_name='features_importance', new_name='feature_importance')
    @docstrings
    def run(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        model: Optional[BasicModel] = None,
        feature_importance: Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: Optional[np.ndarray] = None,
        y_pred_test: Optional[np.ndarray] = None,
        y_proba_train: Optional[np.ndarray] = None,
        y_proba_test: Optional[np.ndarray] = None,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        dataset: Union[Dataset, pd.DataFrame]
            Dataset or DataFrame object, representing data an estimator was fitted on
        model: Optional[BasicModel], default: None
            A scikit-learn-compatible fitted estimator instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train=dataset,
            model=model,
            feature_importance=feature_importance,
            feature_importance_force_permutation=feature_importance_force_permutation,
            feature_importance_timeout=feature_importance_timeout,
            with_display=with_display,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_train=y_proba_train,
            y_proba_test=y_proba_test,
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

    @deprecate_kwarg(old_name='features_importance', new_name='feature_importance')
    @docstrings
    def run(
        self,
        train_dataset: Union[Dataset, pd.DataFrame],
        test_dataset: Union[Dataset, pd.DataFrame],
        model: Optional[BasicModel] = None,
        feature_importance: Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: Optional[np.ndarray] = None,
        y_pred_test: Optional[np.ndarray] = None,
        y_proba_train: Optional[np.ndarray] = None,
        y_proba_test: Optional[np.ndarray] = None,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        train_dataset: Union[Dataset, pd.DataFrame]
            Dataset or DataFrame object, representing data an estimator was fitted on
        test_dataset: Union[Dataset, pd.DataFrame]
            Dataset or DataFrame object, representing data an estimator predicts on
        model: Optional[BasicModel], default: None
            A scikit-learn-compatible fitted estimator instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train=train_dataset,
            test=test_dataset,
            model=model,
            feature_importance=feature_importance,
            feature_importance_force_permutation=feature_importance_force_permutation,
            feature_importance_timeout=feature_importance_timeout,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_train=y_proba_train,
            y_proba_test=y_proba_test,
            with_display=with_display,
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

    @deprecate_kwarg(old_name='features_importance', new_name='feature_importance')
    @docstrings
    def run(
        self,
        model: BasicModel,
        feature_importance: Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: Optional[np.ndarray] = None,
        y_pred_test: Optional[np.ndarray] = None,
        y_proba_train: Optional[np.ndarray] = None,
        y_proba_test: Optional[np.ndarray] = None,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        model: BasicModel
            A scikit-learn-compatible fitted estimator instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None
        context = self.context_type(
            model=model,
            feature_importance=feature_importance,
            feature_importance_force_permutation=feature_importance_force_permutation,
            feature_importance_timeout=feature_importance_timeout,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_train=y_proba_train,
            y_proba_test=y_proba_test,
            with_display=with_display
        )
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

    def run(
        self,
        train_datasets: Union[Dataset, List[Dataset]],
        test_datasets: Union[Dataset, List[Dataset]],
        models: Union[List[BasicModel], Mapping[str, BasicModel]]
    ) -> CheckResult:
        """Initialize context and pass to check logic.

        Parameters
        ----------
        train_datasets: Union[Dataset, List[Dataset]]
            train datasets
        test_datasets: Union[Dataset, List[Dataset]]
            test datasets
        models: Union[List[BasicModel], Mapping[str, BasicModel]]
            list or map of models
        """
        context = ModelComparisonContext(train_datasets, test_datasets, models)
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, multi_context: ModelComparisonContext) -> CheckResult:
        """Implement here logic of check."""
        raise NotImplementedError()
