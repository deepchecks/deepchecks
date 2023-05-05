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
"""Module for base nlp suite."""
# pylint: disable=broad-except
from typing import List, Optional, Tuple, Union

from deepchecks.core import DatasetKind
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.base_checks import SingleDatasetCheck, TrainTestCheck
from deepchecks.nlp.context import Context, TTextPred, TTextProba
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.ipython import create_progress_bar

__all__ = ['Suite']


class Suite(BaseSuite):
    """NLP suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck

    @docstrings
    def run(
        self,
        train_dataset: Union[TextData, None] = None,
        test_dataset: Union[TextData, None] = None,
        with_display: bool = True,
        train_predictions: Optional[TTextPred] = None,
        test_predictions: Optional[TTextPred] = None,
        train_probabilities: Optional[TTextProba] = None,
        test_probabilities: Optional[TTextProba] = None,
        model_classes: Optional[List] = None,
        random_state: int = 42,
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Union[TextData, None] , default: None
            TextData object, representing data an estimator was fitted on
        test_dataset: Union[TextData, None] , default: None
            TextData object, representing data an estimator predicts on
        with_display : bool , default: True
            flag that determines if checks will calculate display (redundant in some checks).
        train_predictions: Union[TTextPred, None] , default: None
            predictions on train dataset
        test_predictions: Union[TTextPred, None] , default: None
            predictions on test dataset
        train_probabilities: Union[TTextProba, None] , default: None
            probabilities on train dataset
        test_probabilities: Union[TTextProba, None] , default: None
            probabilities on test_dataset dataset
        model_classes: Optional[List], default: None
            For classification: list of classes known to the model
        random_state : int, default 42
            A seed to set for pseudo-random functions, primarily sampling.

        {prediction_formats:2*indent}

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = Context(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_pred=train_predictions,
            test_pred=test_predictions,
            train_proba=train_probabilities,
            test_proba=test_probabilities,
            model_classes=model_classes,
            with_display=with_display,
            random_state=random_state
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
                    if train_dataset is not None:
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
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check, exp))

        return SuiteResult(self.name, results)
