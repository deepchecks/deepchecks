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
"""Module for nlp base checks."""
import abc
from typing import List, Optional

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind, SingleDatasetBaseCheck, TrainTestBaseCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.context import Context, TTextPred, TTextProba
from deepchecks.nlp.text_data import TextData

__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
]


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    @docstrings
    def run(
            self,
            dataset: TextData,
            model=None,  # pylint: disable=unused-argument
            with_display: bool = True,
            predictions: Optional[TTextPred] = None,
            probabilities: Optional[TTextProba] = None,
            model_classes: Optional[List] = None,
            random_state: int = 42,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        dataset: TextData
            Dataset representing data an estimator was fitted on
        model: object, default: None
            Dummy model object for compatibility with SingleDatasetBaseCheck
        with_display : bool , default: True
            flag that determines if checks will calculate display (redundant in some checks).
        predictions: Union[TTextPred, None] , default: None
            predictions on dataset
        probabilities: Union[TTextProba, None] , default: None
            probabilities on dataset
        model_classes: Optional[List], default: None
            For classification: list of classes known to the model
        random_state : int, default 42
            A seed to set for pseudo-random functions, primarily sampling.

        {prediction_formats:2*indent}
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train_dataset=dataset,
            with_display=with_display,
            train_pred=predictions,
            train_proba=probabilities,
            model_classes=model_classes,
            random_state=random_state
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

    The class checks train dataset and test_dataset dataset for model training and test_dataset.
    """

    context_type = Context

    @docstrings
    def run(
            self,
            train_dataset: TextData,
            test_dataset: TextData,
            model=None,  # pylint: disable=unused-argument
            with_display: bool = True,
            train_predictions: Optional[TTextPred] = None,
            test_predictions: Optional[TTextPred] = None,
            train_probabilities: Optional[TTextProba] = None,
            test_probabilities: Optional[TTextProba] = None,
            model_classes: Optional[List] = None,
            random_state: int = 42,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        train_dataset: Union[TextData, None] , default: None
            TextData object, representing data an estimator was fitted on
        test_dataset: Union[TextData, None] , default: None
            TextData object, representing data an estimator predicts on
        model: object, default: None
            Dummy model object for compatibility with TrainTestBaseCheck
        with_display : bool , default: True
            flag that determines if checks will calculate display (redundant in some checks).
        train_predictions: Union[TTextPred, None] , default: None
            predictions on train dataset
        test_predictions: Union[TTextPred, None] , default: None
            predictions on test_dataset dataset
        train_probabilities: Union[TTextProba, None] , default: None
            probabilities on train dataset
        test_probabilities: Union[TTextProba, None] , default: None
            probabilities on test_dataset dataset
        model_classes: Optional[List], default: None
            For classification: list of classes known to the model
        random_state : int, default 42
            A seed to set for pseudo-random functions, primarily sampling.

        {prediction_formats:2*indent}
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_pred=train_predictions,
            test_pred=test_predictions,
            train_proba=train_probabilities,
            test_proba=test_probabilities,
            model_classes=model_classes,
            random_state=random_state,
            with_display=with_display,
        )
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()
