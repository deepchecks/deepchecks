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
"""Module for defining metrics for the vision module."""
import typing as t

import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import Precision, Recall, Metric

from torch import nn

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.utils import validation
from deepchecks.vision import VisionDataset

__all__ = [
    'task_type_check',
    'get_scorers_list',
    'calculate_metrics'
]

from .detection_precision_recall import AveragePrecision

from deepchecks.vision.dataset import TaskType


def get_default_classification_scorers():
    return {
        'Precision': Precision(),
        'Recall': Recall()
    }


def get_default_object_detection_scorers():
    return {
        'mAP': AveragePrecision()
    }


def task_type_check(
        model: nn.Module,
        dataset: VisionDataset
) -> TaskType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Parameters
    ----------
    model : nn.Module
        Model object
    dataset : VisionDataset
        Dataset object

    Returns
    -------
    TaskType
        TaskType enum corresponding to the model and dataset
    """
    validation.validate_model(dataset, model)
    dataset.validate_label()

    return TaskType(dataset.label_type)


def get_scorers_list(
        model,
        dataset: VisionDataset,
        alternative_scorers: t.List[Metric] = None,
) -> t.Dict[str, Metric]:
    task_type = task_type_check(model, dataset)

    if alternative_scorers:
        # Validate that each alternative scorer is a correct type
        for met in alternative_scorers:
            if not isinstance(met, Metric):
                raise DeepchecksValueError('alternative_scorers should contain metrics of type ignite.Metric')
        scorers = alternative_scorers
    elif task_type == TaskType.CLASSIFICATION:
        scorers = get_default_classification_scorers()
    elif task_type == TaskType.OBJECT_DETECTION:
        scorers = get_default_object_detection_scorers()
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        scorers = get_default_object_detection_scorers()
    else:
        raise DeepchecksNotSupportedError(f'No scorers match task_type {task_type}')

    return scorers


def validate_prediction(batch_predictions: t.Any, dataset: VisionDataset, eps: float = 1e-3):
    """Validate that the model predictions are in the correct format for working with deepchecks metrics.

    Parameters
    ----------
    batch_predictions : t.Any
        Model prediction for a batch (output of model(batch[0]))
    dataset : VisionDataset
        Dataset object, used only to get label_type and n_classes
    eps : float, optional
        Epsilon value to be used in the validation, by default 1e-3
    """

    label_type = dataset.label_type
    n_classes = dataset.get_num_classes()

    if label_type == TaskType.CLASSIFICATION.value:
        if not isinstance(batch_predictions, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a torch.Tensor or numpy '
                                       f'array')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a 2D tensor')
        if pred_shape[1] != n_classes:
            raise DeepchecksValueError(f'Check requires {label_type} predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(axis=1) - 1) > eps):
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a probability distribution and'
                                       f' sum to 1 for each row')
    elif dataset.label_type == TaskType.OBJECT_DETECTION.value:
        if not isinstance(batch_predictions, list):
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a list with an entry for each'
                                       f' sample')
        if len(batch_predictions) == 0:
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a non-empty list')
        if not isinstance(batch_predictions[0], (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a list of torch.Tensor or'
                                       f' numpy array')
        if len(batch_predictions[0].shape) != 2:
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a list of 2D tensors')
        if batch_predictions[0].shape[1] != 6:
            raise DeepchecksValueError(f'Check requires {label_type} predictions to be a list of 2D tensors, when '
                                       f'each row has 6 columns: [x, y, width, height, class_probability, class_id]')
    else:
        raise NotImplementedError(
            'Not implemented yet for tasks other than classification and object detection'
        )


def calculate_metrics(metrics: t.List[Metric], dataset: VisionDataset, model: nn.Module,
                      prediction_extract: t.Callable = None) -> t.Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Parameters
    ----------
    metrics : t.List[Metric]
        List of ignite metrics to calculate
    dataset : VisionDataset
        Dataset object
    model : nn.Module
        Model object
    prediction_extract : t.Callable
        Function to convert the model output to the appropriate format for the label type

    Returns
    -------
    t.Dict[str, float]
        Dictionary of metrics with the metric name as key and the metric value as value
    """

    def process_function(_, batch):
        images = batch[0]
        label = dataset.label_transformer(batch[1])

        predictions = model.forward(images)

        if prediction_extract:
            predictions = prediction_extract(predictions)

        return predictions, label

    # Validate that
    data_batch = process_function(None, next(iter(dataset)))[0]
    validate_prediction(data_batch, dataset)

    engine = Engine(process_function)
    for metric in metrics:
        metric.attach(engine, type(metric).__name__)

    state = engine.run(dataset.get_data_loader())

    results = {k: v.tolist() for k, v in state.metrics.items()}
    return results
