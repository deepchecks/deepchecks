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
"""The confusion_matrix_report check module."""
import numpy as np
import pandas as pd
from sklearn import metrics

from deepchecks.core import CheckResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.plot import create_confusion_matrix_figure

__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetCheck):
    """Calculate the confusion matrix of the model on the given dataset.

    Parameters
    ----------
    normalized (bool, default True):
        boolean that determines whether to normalize the true values of the matrix.
    """

    def __init__(self,
                 normalized: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.normalized = normalized

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
        dataset = context.get_data_by_kind(dataset_kind)
        context.assert_classification_task()
        ds_y = dataset.label_col
        ds_x = dataset.features_columns
        model = context.model

        y_pred = np.array(model.predict(ds_x)).reshape(len(ds_y), )
        total_classes = sorted(list(set(pd.concat([ds_y, pd.Series(y_pred)]).to_list())))
        confusion_matrix = metrics.confusion_matrix(ds_y, y_pred)

        if context.with_display:
            fig = create_confusion_matrix_figure(confusion_matrix, total_classes,
                                                 total_classes, self.normalized)
        else:
            fig = None

        return CheckResult(confusion_matrix, display=fig)
