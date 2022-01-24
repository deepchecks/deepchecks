# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The confusion_matrix_report check module."""
import sklearn

import plotly.express as px

from deepchecks.base.check_context import CheckRunContext
from deepchecks import CheckResult
from deepchecks.base.check import SingleDatasetBaseCheck


__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Calculate the confusion matrix of the model on the given dataset."""

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
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
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        context.assert_classification_task()
        ds_y = dataset.data[context.label_name]
        ds_x = dataset.data[context.features]
        model = context.model

        y_pred = model.predict(ds_x)
        confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

        # Figure
        fig = px.imshow(confusion_matrix, x=dataset.classes, y=dataset.classes, text_auto=True)
        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value', type='category')
        fig.update_yaxes(title='True value', type='category')

        return CheckResult(confusion_matrix, display=fig)
