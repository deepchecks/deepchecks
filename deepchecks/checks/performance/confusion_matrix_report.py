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
from sklearn.base import BaseEstimator

import plotly.express as px
from deepchecks import CheckResult, Dataset
from deepchecks.base.check import SingleDatasetBaseCheck
from deepchecks.utils.metrics import ModelType


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
        dataset = Dataset.ensure_not_empty_dataset(dataset)
        ds_y = self._dataset_has_label(dataset)
        ds_x = self._dataset_has_features(dataset)

        self._verify_model_type(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        y_pred = model.predict(ds_x)
        confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

        # Figure
        fig = px.imshow(confusion_matrix, x=dataset.classes, y=dataset.classes, text_auto=True)
        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value', type='category')
        fig.update_yaxes(title='True value', type='category')

        return CheckResult(confusion_matrix, display=fig)
