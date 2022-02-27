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

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError

from deepchecks.vision.dataset import TaskType
from deepchecks.vision.utils.base_formatters import BasePredictionFormatter
from deepchecks.vision import VisionData
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision


__all__ = [
    'get_scorers_list',
    'calculate_metrics',
    'metric_results_to_df',
    'filter_classes_for_display',
]


def get_default_classification_scorers():
    return {
        'Precision': Precision(),
        'Recall': Recall()
    }


def get_default_object_detection_scorers():
    return {
        'AP': AveragePrecision(),
        'AR': AveragePrecision(return_option=1)
    }


def get_scorers_list(
        dataset: VisionData,
        alternative_scorers: t.Dict[str, Metric] = None
) -> t.Dict[str, Metric]:
    """Get scorers list according to model object and label column.

    Parameters
    ----------
    dataset : VisionData
        Dataset object
    alternative_scorers : t.Dict[str, Metric]
        Alternative scorers dictionary
    Returns
    -------
    t.Dict[str, Metric]
        Scorers list
    """
    task_type = dataset.task_type

    if alternative_scorers:
        # Validate that each alternative scorer is a correct type
        for _, met in alternative_scorers.items():
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
    metrics: t.Dict[str, Metric],
    dataset: VisionData,
    model: nn.Module,
    prediction_formatter: BasePredictionFormatter,
    device: torch.device
) -> t.Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Parameters
    ----------
    metrics : Dict[str, Metric]
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
        return prediction_formatter(batch, model, device), dataset.label_formatter(batch)

    engine = Engine(process_function)

    for name, metric in metrics.items():
        metric.reset()
        metric.attach(engine, name)

    state = engine.run(dataset.get_data_loader())
    return state.metrics


def metric_results_to_df(results: dict, dataset: VisionData) -> pd.DataFrame:
    """Get dict of metric name to tensor of classes scores, and convert it to dataframe."""
    per_class_result = [
        [metric, class_id, dataset.label_id_to_name(class_id),
         class_score.item() if isinstance(class_score, torch.Tensor) else class_score]
        for metric, score in results.items()
        # scorer returns results as array, containing result per class
        for class_score, class_id in zip(score, sorted(dataset.n_of_samples_per_class.keys()))
    ]

    return pd.DataFrame(per_class_result, columns=['Metric',
                                                   'Class',
                                                   'Class Name',
                                                   'Value']).sort_values(by=['Metric', 'Class'])


def filter_classes_for_display(metrics_df: pd.DataFrame,
                               metric_to_show_by: str,
                               n_to_show: int,
                               show_only: str) -> list:
    """Filter the metrics dataframe for display purposes.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing the metrics.
    n_to_show : int
        Number of classes to show in the report.
    show_only : str, default: 'largest'
        Specify which classes to show in the report. Can be one of the following:
        - 'largest': Show the largest classes.
        - 'smallest': Show the smallest classes.
        - 'random': Show random classes.
        - 'best': Show the classes with the highest score.
        - 'worst': Show the classes with the lowest score.
    metric_to_show_by : str
        Specify the metric to sort the results by. Relevant only when show_only is 'best' or 'worst'.

    Returns
    -------
    list
        List of classes to show in the report.
    """
    # working only on the test set
    tests_metrics_df = metrics_df[(metrics_df['Dataset'] == DatasetKind.TEST.value) &
                                  (metrics_df['Metric'] == metric_to_show_by)]
    print(tests_metrics_df)
    if show_only == 'largest':
        tests_metrics_df = tests_metrics_df.sort_values(by='Number of samples', ascending=False)
    elif show_only == 'smallest':
        tests_metrics_df = tests_metrics_df.sort_values(by='Number of samples', ascending=True)
    elif show_only == 'random':
        tests_metrics_df = tests_metrics_df.sample(frac=1)
    elif show_only == 'best':
        tests_metrics_df = tests_metrics_df.sort_values(by='Value', ascending=False)
    elif show_only == 'worst':
        tests_metrics_df = tests_metrics_df.sort_values(by='Value', ascending=True)
    else:
        raise ValueError(f'Unknown show_only value: {show_only}')

    return tests_metrics_df.head(n_to_show)['Class'].to_list()
