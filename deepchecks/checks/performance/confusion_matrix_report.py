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
import numpy as np
import sklearn
from sklearn.base import BaseEstimator

import plotly.figure_factory as ff
from deepchecks import CheckResult, Dataset
from deepchecks.base.check import SingleDatasetBaseCheck
from deepchecks.utils.metrics import ModelType, task_type_validation


__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Calculate the confusion matrix of the model on the given dataset."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            dataset: a Dataset object

        Returns:
            CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._confusion_matrix_report(dataset, model)

    def _confusion_matrix_report(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        task_type_validation(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        label = dataset.label_name
        ds_x = dataset.data[dataset.features]
        ds_y = dataset.data[label]
        y_pred = model.predict(ds_x)

        confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

        labels = [str(val) for val in np.unique(ds_y)]
        fig = ff.create_annotated_heatmap(confusion_matrix, x=labels, y=labels, colorscale='Viridis')
        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value')
        fig.update_yaxes(title='True value', autorange='reversed')
        fig['data'][0]['showscale'] = True
        fig['layout']['xaxis']['side'] = 'bottom'

        return CheckResult(confusion_matrix, display=fig)
