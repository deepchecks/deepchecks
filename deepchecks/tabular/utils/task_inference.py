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
"""Utils module containing functionalities to infer the task type and possible label classes."""
from typing import Optional

import numpy as np
import pandas as pd
from pandas._libs.lib import infer_dtype

from deepchecks.core.errors import ValidationError
from deepchecks.tabular.utils.feature_inference import is_categorical
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.array_math import sequence_to_numpy
from deepchecks.utils.logger import get_logger
from deepchecks.utils.typing import BasicModel

__all__ = ['infer_task_type_by_labels', 'infer_classes_from_model', 'get_all_labels', 'infer_task_type_by_class_number']


def infer_classes_from_model(model: Optional[BasicModel]):
    """Get classes_ attribute from model object if exists."""
    if model is not None and hasattr(model, 'classes_') and len(model.classes_) > 0:
        return sorted(list(model.classes_))


def get_all_labels(model, train_dataset, test_dataset=None, y_pred_train=None, y_pred_test=None):
    """Aggregate labels from all available data: labels on datasets, y_pred, and model predicitions."""
    labels = np.asarray([])
    if train_dataset:
        if train_dataset.has_label():
            labels = np.append(labels, train_dataset.label_col.to_numpy())
        if model:
            labels = np.append(labels, sequence_to_numpy(model.predict(train_dataset.features_columns)))
    if test_dataset:
        if test_dataset.has_label():
            labels = np.append(labels, test_dataset.label_col.to_numpy())
        if model:
            labels = np.append(labels, sequence_to_numpy(model.predict(test_dataset.features_columns)))
    if y_pred_train is not None:
        labels = np.append(labels, y_pred_train)
    if y_pred_test is not None:
        labels = np.append(labels, y_pred_test)

    return pd.Series(labels) if len(labels) > 0 else pd.Series(dtype='object')


def infer_task_type_by_labels(labels: pd.Series):
    """Infer task type from given dataset/labels/model_classes."""
    # there are no observed labels (user didn't pass model, and datasets have no label column), then we
    # have no task type
    if len(labels) == 0:
        return None
    # Fourth, we check if the observed labels are categorical or not
    if is_categorical(labels, max_categorical_ratio=0.05):
        num_classes = len(labels.dropna().unique())
        task_type = infer_task_type_by_class_number(num_classes)
        if infer_dtype(labels) == 'integer':
            get_logger().warning(
                'Due to the small number of unique labels task type was inferred as %s classification in spite of '
                'the label column is of type integer. '
                'Initialize your Dataset with either label_type=\"%s\" or '
                'label_type=\"regression\" to resolve this warning.', task_type.value, task_type.value)
        return task_type
    else:
        return TaskType.REGRESSION


def infer_task_type_by_class_number(num_classes):
    """Infer task type of binary or multiclass."""
    if num_classes == 0:
        raise ValidationError('Found zero number of classes')
    if num_classes == 1:
        raise ValidationError('Found only one class in label column, pass the full list of possible '
                              'label classes via the model_classes argument of the run function.')
    return TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
