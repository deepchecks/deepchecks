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
"""Module for base tabular abstractions."""
# pylint: disable=broad-except
from typing import Optional, Tuple, Union

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
from deepchecks.utils.decorators import deprecate_kwarg
from deepchecks.utils.ipython import create_progress_bar
from deepchecks.utils.typing import BasicModel

__all__ = ['Suite']


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck

    @deprecate_kwarg(old_name='features_importance', new_name='feature_importance')
    @docstrings
    def run(
        self,
        train_dataset: Union[Dataset, pd.DataFrame, None] = None,
        test_dataset: Union[Dataset, pd.DataFrame, None] = None,
        model: Optional[BasicModel] = None,
        feature_importance: Optional[pd.Series] = None,
        feature_importance_force_permutation: bool = False,
        feature_importance_timeout: int = 120,
        with_display: bool = True,
        y_pred_train: Optional[np.ndarray] = None,
        y_pred_test: Optional[np.ndarray] = None,
        y_proba_train: Optional[np.ndarray] = None,
        y_proba_test: Optional[np.ndarray] = None,
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator was fitted on
        test_dataset : Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator predicts on
        model : Optional[BasicModel] , default None
            A scikit-learn-compatible fitted estimator instance
        {additional_context_params:2*indent}

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = Context(
            train_dataset,
            test_dataset,
            model,
            feature_importance=feature_importance,
            feature_importance_force_permutation=feature_importance_force_permutation,
            feature_importance_timeout=feature_importance_timeout,
            with_display=with_display,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_train=y_proba_train,
            y_proba_test=y_proba_test,
        )

        progress_bar = create_progress_bar(
            iterable=list(self.checks.values()),
            name=self.name,
            unit='Check'
        )

        # Run all checks
        results = []
        for check in progress_bar:
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
                    if train_dataset is not None:
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
                    if test_dataset is not None:
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

        return SuiteResult(self.name, results)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
