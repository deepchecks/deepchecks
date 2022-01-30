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
    """Check task type (classification, object-detection, semantic-segmentation) according to model object and label \
    column.

    Parameters
    ----------
    model : nn.Module
        Model object
    dataset : VisionDataset
        Dataset object

    Returns
    -------
    TaskType
        enum corresponding to the model and dataset
    """
    validation.model_type_validation(model)
    dataset.assert_label()

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

    engine = Engine(process_function)
    for metric in metrics:
        metric.attach(engine, type(metric).__name__)

    state = engine.run(dataset.get_data_loader())

    results = {k: v.tolist() for k, v in state.metrics.items()}
    return results
