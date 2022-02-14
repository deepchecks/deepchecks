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

import pandas as pd
import torch
from ignite.engine import Engine
from ignite.metrics import Precision, Recall, Metric
from torch import nn

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError

from deepchecks.vision.dataset import TaskType
from deepchecks.vision.utils.base_formatters import BasePredictionFormatter
from deepchecks.vision import VisionData
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision


__all__ = [
    'get_scorers_list',
    'calculate_metrics',
    'metric_results_to_df'
]


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
        dataset: VisionData,
        alternative_scorers: t.List[Metric] = None
) -> t.Dict[str, Metric]:
    """Get scorers list according to model object and label column.

    Parameters
    ----------
    dataset : VisionData
        Dataset object
    alternative_scorers : t.List[Metric]
        Alternative scorers list
    class_average : bool, default: False
        Whether classification metrics should average the results or return result per class
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


def calculate_metrics(
    metrics: t.Union[t.Dict, t.List[Metric]],
    dataset: VisionData, model: nn.Module,
    prediction_formatter: BasePredictionFormatter,
    device: t.Union[str, torch.device, None] = None
) -> t.Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Parameters
    ----------
    metrics : List[Metric]
        List of ignite metrics to calculate
    dataset : VisionData
        Dataset object

    model : nn.Module
        Model object
    prediction_formatter : Union[ClassificationPredictionFormatter, DetectionPredictionFormatter]
        Function to convert the model output to the appropriate format for the label type
    device : Union[str, torch.device, None]

    Returns
    -------
    t.Dict[str, float]
        Dictionary of metrics with the metric name as key and the metric value as value
    """

    def process_function(_, batch):
        images = batch[0]
        label = dataset.label_transformer(batch[1])

        if isinstance(images, torch.Tensor):
            images = images.to(device)
        if isinstance(label, torch.Tensor):
            label = label.to(device)

        predictions = model.forward(images)

        if prediction_formatter:
            predictions = prediction_formatter(predictions)

        return predictions, label

    # Validate that
    data_batch = process_function(None, next(iter(dataset)))[0]
    prediction_formatter.validate_prediction(data_batch, dataset.n_of_classes)

    engine = Engine(process_function)

    if isinstance(metrics, list):
        metrics = {type(metric).__name__: metric for metric in metrics}

    for name, metric in metrics.items():
        metric.reset()
        metric.attach(engine, name)

    state = engine.run(dataset.get_data_loader())
    return state.metrics


def metric_results_to_df(results: dict, dataset: VisionData) -> pd.DataFrame:
    """Get dict of metric name to tensor of classes scores, and convert it to dataframe."""
    per_class_result = [
        [metric, class_name,
         class_score.item() if isinstance(class_score, torch.Tensor) else class_score]
        for metric, score in results.items()
        # scorer returns results as array, containing result per class
        for class_score, class_name in zip(score, sorted(dataset.get_samples_per_class().keys()))
    ]

    return pd.DataFrame(per_class_result, columns=['Metric', 'Class', 'Value']).sort_values(by=['Metric', 'Class'])
