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
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.task_type import TaskType
from deepchecks.utils.abstracts.confusion_matrix_abstract import run_confusion_matrix_check

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
        type_name = type(self).__name__

        if context.task_type is TaskType.TOKEN_CLASSIFICATION:
            task_type_name = TaskType.TOKEN_CLASSIFICATION.value
            raise DeepchecksNotSupportedError(f'"{type_name}" is not suited for the "{task_type_name}" tasks')

        dataset = context.get_data_by_kind(dataset_kind)

        if dataset.is_multi_label_classification():
            raise DeepchecksNotSupportedError(f'"{type_name}" is not suited for the multilable classification tasks')

        dataset = dataset.sample(self.n_samples, random_state=self.random_state)
        y_true = np.asarray(dataset.label)
        y_pred = np.array(context.model.predict(dataset)).reshape(len(y_true), )

        return run_confusion_matrix_check(y_pred, y_true, context.with_display, self.normalize_display)
