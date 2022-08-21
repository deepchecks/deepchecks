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
"""Module for nlp base checks."""
import abc
from typing import Optional


from deepchecks.nlp.context import TTextPred, Context
from deepchecks.nlp.text_data import TextData

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import (DatasetKind, SingleDatasetBaseCheck,
                                    TrainTestBaseCheck)


__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
]


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    def run(
        self,
        dataset: TextData,
        model=None,  # pylint: disable=unused-argument
        with_display: bool = True,
        predictions: Optional[TTextPred] = None,
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
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train_dataset=dataset,
            with_display=with_display,
            train_predictions=predictions
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

    def run(
        self,
        train_dataset: TextData,
        test_dataset: TextData,
        model=None,  # pylint: disable=unused-argument
        with_display: bool = True,
        train_predictions: Optional[TTextPred] = None,
        test_predictions: Optional[TTextPred] = None,
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
        """
        assert self.context_type is not None
        context = self.context_type(  # pylint: disable=not-callable
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            with_display=with_display,
        )
        result = self.run_logic(context)
        context.finalize_check_result(result, self)
        return result

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()

