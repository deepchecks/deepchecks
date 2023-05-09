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
"""The confusion_matrix_report check module."""
import numpy as np

from deepchecks.core import CheckResult
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.utils.abstracts.confusion_matrix_abstract import (misclassified_samples_lower_than_condition,
                                                                  run_confusion_matrix_check)
from deepchecks.utils.strings import format_percent

__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetCheck):
    """Calculate the confusion matrix of the model on the given dataset.

    Parameters
    ----------
    normalize_display : bool , default: True:
        boolean that determines whether to normalize the values of the matrix in the display.
    n_samples : int , default: 10_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 normalize_display: bool = True,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.normalize_display = normalize_display
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is numpy array of the confusion matrix, displays the confusion matrix

        Raises
        ------
        DeepchecksValueError
            If the data is not a Dataset instance with a label
        """
        context.raise_if_token_classification_task(self)
        context.raise_if_multi_label_task(self)

        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.sample(self.n_samples, random_state=self.random_state)
        y_true = np.asarray(dataset.label)
        y_pred = np.array(context.model.predict(dataset)).reshape(len(y_true), )

        return run_confusion_matrix_check(y_pred, y_true, context.with_display, self.normalize_display)

    def add_condition_misclassified_samples_lower_than_condition(self, misclassified_samples_threshold: float = 0.2):
        """Add condition - Misclassified samples lower than threshold.

        Condition validates if the misclassified cell size/samples are lower than the threshold based on the
        `misclassified_samples_threshold` parameter.

        Parameters
        ----------
        misclassified_samples_threshold: float, default: 0.20
            Ratio of samples to be used for comparison in the condition (Value should be between 0 - 1 inclusive)
        """
        return self.add_condition(
            f'Misclassified cell size lower than {format_percent(misclassified_samples_threshold)} '
            'of the total samples',
            misclassified_samples_lower_than_condition,
            misclassified_samples_threshold=misclassified_samples_threshold
        )
