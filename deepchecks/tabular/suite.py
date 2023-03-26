# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for base tabular abstractions."""
# pylint: disable=broad-except
import time
import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.utils.ipython import create_progress_bar
from deepchecks.utils.typing import BasicModel

if TYPE_CHECKING:
    from deepchecks.recommender.dataset import RecDataset, ItemDataset

__all__ = ['Suite']


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> t.Tuple:
        """Return t.Tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck

    context_type = None

    @t.overload
    @docstrings
    def run(
        self,
        train_dataset: t.Union[Dataset, pd.DataFrame, None] = None,
        test_dataset: t.Union[Dataset, pd.DataFrame, None] = None,
        model: t.Optional[BasicModel] = None,
        feature_importance: t.Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: t.Optional[np.ndarray] = None,
        y_pred_test: t.Optional[np.ndarray] = None,
        y_proba_train: t.Optional[np.ndarray] = None,
        y_proba_test: t.Optional[np.ndarray] = None,
        run_single_dataset: t.Optional[str] = None,
        model_classes: t.Optional[t.List] = None,
        item_dataset=None,
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: t.Optional[t.Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator was fitted on
        test_dataset : t.Optional[t.Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator predicts on
        model : t.Optional[BasicModel] , default None
            A scikit-learn-compatible fitted estimator instance
        run_single_dataset: t.Optional[str], default None
            'Train', 'Test' , or None to run on both train and test.
        {additional_context_params:2*indent}

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
    
    @t.overload
    @docstrings
    def run(
        self,
        train_dataset: t.Optional['RecDataset']  = None,
        test_dataset: t.Optional['RecDataset']  = None,
        model: t.Optional[BasicModel] = None,
        feature_importance: t.Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: t.Optional[np.ndarray] = None,
        y_pred_test: t.Optional[np.ndarray] = None,
        run_single_dataset: t.Optional[str] = None,
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train: Optional[RecDataset] , default: None
            RecDataset object (dataset object for recommendation systems), representing data an estimator was fitted on
        test: Optional[RecDataset] , default: None
            RecDataset object (dataset object for recommendation systems), representing data an estimator was fitted on
        feature_importance: pd.Series , default: None
            pass manual features importance
        feature_importance_force_permutation : bool , default: False
            force calculation of permutation features importance
        feature_importance_timeout : int , default: 120
            timeout in second for the permutation features importance calculation
        y_pred_train: Optional[np.ndarray] , default: None
            Array of the model prediction over the train dataset.
        y_pred_test: Optional[np.ndarray] , default: None
            Array of the model prediction over the test dataset.
        run_single_dataset: t.Optional[str], default None
            'Train', 'Test' , or None to run on both train and test.

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """

    def run(
        self,
        train_dataset: t.Union[Dataset, pd.DataFrame, None] = None,
        test_dataset: t.Union[Dataset, pd.DataFrame, None] = None,
        model: t.Optional[BasicModel] = None,
        item_dataset: t.Optional['ItemDataset'] = None,
        feature_importance: t.Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: t.Optional[np.ndarray] = None,
        y_pred_test: t.Optional[np.ndarray] = None,
        y_proba_train: t.Optional[np.ndarray] = None,
        y_proba_test: t.Optional[np.ndarray] = None,
        run_single_dataset: t.Optional[str] = None,
        model_classes: t.Optional[t.List] = None,
    ) -> SuiteResult:
        from deepchecks.recommender.context import Context as RecContext
        from deepchecks.recommender.dataset import RecDataset
        
        if not isinstance(train_dataset, RecDataset) and item_dataset is not None:
            raise DeepchecksNotSupportedError('item_dataset is not supported for tabular datasets.')

        if isinstance(train_dataset, RecDataset) and model_classes is not None:
            raise DeepchecksNotSupportedError('model_classes is not supported for recommendation datasets.')
    
        if self.context_type is None:
            if isinstance(train_dataset, RecDataset):
                self.context_type = RecContext
                context = self.context_type(  # pylint: disable=not-callable
                    train=train_dataset,
                    test=test_dataset,
                    item_dataset=item_dataset,
                    feature_importance=feature_importance,
                    feature_importance_force_permutation=feature_importance_force_permutation,
                    feature_importance_timeout=feature_importance_timeout,
                    y_pred_train=y_pred_train,
                    y_pred_test=y_pred_test,
                    with_display=with_display,
                )
            else:
                self.context_type = Context
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
                    model_classes=model_classes
                )

        progress_bar = create_progress_bar(
            iterable=t.List(self.checks.values()),
            name=self.name,
            unit='Check'
        )

        # Run all checks
        results = []
        for check in progress_bar:
            start = time.time()

            try:
                progress_bar.set_postfix({'Check': check.name()}, refresh=False)
                if isinstance(check, TrainTestCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run_logic(context)
                        context.finalize_check_result(check_result, check)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if not supplied with both train and test datasets'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, SingleDatasetCheck):
                    if train_dataset is not None and (run_single_dataset in [DatasetKind.TRAIN.value, None]):
                        # In case of train & test, doesn't want to skip test if train fails. so have to explicitly
                        # wrap it in try/except
                        try:
                            check_result = check.run_logic(context, dataset_kind=DatasetKind.TRAIN)
                            context.finalize_check_result(check_result, check, DatasetKind.TRAIN)
                            # In case of single dataset not need to edit the header
                            if test_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Train Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Train Dataset')
                        results.append(check_result)
                    if test_dataset is not None and (run_single_dataset in [DatasetKind.TEST.value, None]):
                        try:
                            check_result = check.run_logic(context, dataset_kind=DatasetKind.TEST)
                            context.finalize_check_result(check_result, check, DatasetKind.TEST)
                            # In case of single dataset not need to edit the header
                            if train_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Test Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Test Dataset')
                        results.append(check_result)
                    if train_dataset is None and test_dataset is None:
                        msg = 'Check is irrelevant if dataset is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, ModelOnlyCheck):
                    if model is not None:
                        check_result = check.run_logic(context)
                        context.finalize_check_result(check_result, check)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if model is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check, exp))

            results[-1].run_time = int(round(time.time() - start, 0))

        return SuiteResult(self.name, results)
