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
"""Module for defining metrics for the vision module."""
import typing as t

from ignite.engine import Engine
from ignite.metrics import Precision, Recall, Metric

from torch import nn

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision import VisionDataset

__all__ = [
    'get_scorers_list',
    'calculate_metrics'
]

from .detection_precision_recall import AveragePrecision

from deepchecks.vision.dataset import TaskType
from deepchecks.vision.utils.base_encoders import BasePredictionEncoder


def get_default_classification_scorers():
    return {
        'Precision': Precision(),
        'Recall': Recall()
    }


def get_default_object_detection_scorers():
    return {
        'mAP': AveragePrecision()
    }


def get_scorers_list(
        dataset: VisionDataset,
        alternative_scorers: t.List[Metric] = None,
) -> t.Dict[str, Metric]:
    """Get scorers list according to model object and label column.

    Parameters
    ----------
    dataset : VisionDataset
        Dataset object
    alternative_scorers : t.List[Metric]
        Alternative scorers list

    Returns
    -------
    t.Dict[str, Metric]
        Scorers list
    """
    task_type = dataset.task_type

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
                      prediction_encoder: BasePredictionEncoder) \
        -> t.Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Parameters
    ----------
    metrics : t.List[Metric]
        List of ignite metrics to calculate
    dataset : VisionDataset
        Dataset object
    model : nn.Module
        Model object
    prediction_encoder : Union[ClassificationPredictionEncoder, DetectionPredictionEncoder]
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

        if prediction_encoder:
            predictions = prediction_encoder(predictions)

        return predictions, label

    # Validate that
    data_batch = process_function(None, next(iter(dataset)))[0]
    prediction_encoder.validate_prediction(data_batch, dataset.get_num_classes())

    engine = Engine(process_function)
    for metric in metrics:
        metric.attach(engine, type(metric).__name__)

    state = engine.run(dataset.get_data_loader())

    results = {k: v.tolist() for k, v in state.metrics.items()}
    return results
