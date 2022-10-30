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

__all__ = ['infer_task_type_and_classes']

from typing import List, Optional, Tuple

import pandas as pd
from pandas._libs.lib import infer_dtype

from deepchecks import tabular  # pylint: disable=unused-import; it is used for type annotations
from deepchecks.core.errors import ValidationError
from deepchecks.tabular.utils.feature_inference import is_categorical
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.array_math import convert_into_flat_list
from deepchecks.utils.logger import get_logger
from deepchecks.utils.typing import BasicModel


def infer_task_type_and_classes(model: Optional[BasicModel], train_dataset: 'tabular.Dataset',
                                test_dataset: Optional['tabular.Dataset'] = None,
                                model_classes: Optional[List] = None) -> \
        Tuple[TaskType, Optional[List], Optional[List]]:
    """Infer the task type based on labels in the data and the model. For classification also computes the classes in \
    the model and the observed classes.

    Parameters
    ----------
    model : BasicModel
        Model object used in task
    train_dataset : 'tabular.Dataset'
        Train Dataset of task
    test_dataset : Optional['tabular.Dataset'], default = None
        Test Dataset of task
    model_classes
        Model's classes if provided by the user manually.

    Returns
    -------
    (TaskType, List, List)
        The type of the Task, The observed classes, The model classes
    """
    train_labels = []
    test_labels = []
    have_model = model is not None
    if train_dataset:
        if train_dataset.has_label():
            train_labels += train_dataset.label_col.to_list()
        if have_model:
            train_labels += convert_into_flat_list(model.predict(train_dataset.features_columns))
    if test_dataset:
        if test_dataset.has_label():
            test_labels += test_dataset.label_col.to_list()
        if have_model:
            test_labels += convert_into_flat_list(model.predict(test_dataset.features_columns))

    observed_labels = pd.Series(test_labels + train_labels)
    if model_classes is None and have_model and hasattr(model, 'classes_') and len(model.classes_) > 0:
        model_classes = sorted(list(model.classes_))

    if train_dataset and train_dataset.label_type is not None:
        task_type = train_dataset.label_type
    elif model_classes:
        task_type = infer_by_class_number(len(model_classes))
    elif len(observed_labels) > 0 and is_categorical(observed_labels, max_categorical_ratio=0.05):
        num_classes = len(observed_labels.dropna().unique())
        task_type = infer_by_class_number(num_classes)
        if infer_dtype(observed_labels) == 'integer' and train_dataset and train_dataset.label_type is None:
            get_logger().warning(
                'Due to the small number of unique labels task type was inferred as classification in spite of '
                'the label column is of type integer. '
                'Initialize your Dataset with either label_type=\"multiclass\" or '
                'label_type=\"regression\" to resolve this warning.')
    else:
        task_type = TaskType.REGRESSION

    if task_type in [TaskType.BINARY, TaskType.MULTICLASS]:
        return task_type, sorted(observed_labels.dropna().unique()), model_classes
    else:
        return task_type, None, None


def infer_by_class_number(num_classes):
    if num_classes == 0:
        raise ValidationError('Found zero number of classes')
    if num_classes == 1:
        raise ValidationError('Found only one class in label column, pass the full list of possible '
                              'label classes via the model_classes argument of the run function.')
    return TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
