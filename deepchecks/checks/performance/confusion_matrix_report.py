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

        def display():
            sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()

        return CheckResult(confusion_matrix, display=display)

