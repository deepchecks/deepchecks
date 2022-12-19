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

__all__ = ['get_labels_and_classes', 'infer_task_type', 'infer_task_type_and_classes']

from typing import List, Optional

import pandas as pd
from pandas._libs.lib import infer_dtype

from deepchecks import tabular  # pylint: disable=unused-import; it is used for type annotations
from deepchecks.core.errors import ValidationError
from deepchecks.tabular.utils.feature_inference import is_categorical
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.array_math import convert_into_flat_list
from deepchecks.utils.logger import get_logger
from deepchecks.utils.typing import BasicModel


def get_labels_and_classes(
        model: Optional[BasicModel], train_dataset: 'tabular.Dataset',
        test_dataset: Optional['tabular.Dataset'] = None, y_pred_train=None, y_pred_test=None,
        model_classes: Optional[List] = None):
    """Get the classes of the model and the observed labels."""
    labels = []
    have_model = model is not None
    if train_dataset:
        if train_dataset.has_label():
            labels += train_dataset.label_col.to_list()
        if have_model:
            labels += convert_into_flat_list(model.predict(train_dataset.features_columns))
    if test_dataset:
        if test_dataset.has_label():
            labels += test_dataset.label_col.to_list()
        if have_model:
            labels += convert_into_flat_list(model.predict(test_dataset.features_columns))
    if y_pred_train is not None:
        labels += convert_into_flat_list(y_pred_train)
    if y_pred_test is not None:
        labels += convert_into_flat_list(y_pred_test)

    labels = pd.Series(labels) if len(labels) > 0 else pd.Series(dtype='object')
    if model_classes is None and have_model and hasattr(model, 'classes_') and len(model.classes_) > 0:
        model_classes = sorted(list(model.classes_))
    return labels, model_classes


def infer_task_type(train_dataset: 'tabular.Dataset', labels, model_classes: Optional[List]):
    """Infer the task type based on dataset, observed labels and model classes."""
    # First if the user defined manually the task type (label type on dataset) we use it
    if train_dataset and train_dataset.label_type is not None:
        return train_dataset.label_type
    # Secondly if user passed model_classes or we found classes on the model object, we use them
    elif model_classes:
        return infer_by_class_number(len(model_classes))
    # Thirdly if there are no observed labels (user didn't pass model, and datasets have no label column), then we
    # have no task type
    elif len(labels) == 0:
        return None
    # Fourth, we check if the observed labels are categorical or not
    elif is_categorical(labels, max_categorical_ratio=0.05):
        num_classes = len(labels.dropna().unique())
        task_type = infer_by_class_number(num_classes)
        if infer_dtype(labels) == 'integer' and train_dataset and train_dataset.label_type is None:
            get_logger().warning(
                'Due to the small number of unique labels task type was inferred as %s classification in spite of '
                'the label column is of type integer. '
                'Initialize your Dataset with either label_type=\"%s}\" or '
                'label_type=\"regression\" to resolve this warning.', task_type.value, task_type.value)
    else:
        return TaskType.REGRESSION


def infer_task_type_and_classes(model, dataset):
    """Doing both classes inference and task type inference."""
    labels, model_classes = get_labels_and_classes(model, dataset)
    task_type = infer_task_type(dataset, labels, model_classes)
    observed_classes = sorted(list(set(labels)))
    return task_type, observed_classes, model_classes


def infer_by_class_number(num_classes):
    if num_classes == 0:
        raise ValidationError('Found zero number of classes')
    if num_classes == 1:
        raise ValidationError('Found only one class in label column, pass the full list of possible '
                              'label classes via the model_classes argument of the run function.')
    return TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
