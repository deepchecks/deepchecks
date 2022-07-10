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
from copy import copy

import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine
from ignite.metrics import Metric, Precision, Recall
from sklearn.metrics._scorer import _BaseScorer, _ProbaScorer

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.vision.metrics_utils import (AVAILABLE_EVALUTING_FUNCTIONS, CustomScorer,
                                             ObjectDetectionAveragePrecision, ObjectDetectionTpFpFn)
from deepchecks.vision.vision_data import TaskType, VisionData

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


def get_default_object_detection_scorers() -> t.Dict[str, Metric]:
    return {
        'Average Precision': ObjectDetectionAveragePrecision(return_option='ap'),
        'Average Recall': ObjectDetectionAveragePrecision(return_option='ar')
    }


_func_naming_dict = {
    'precision': 'Precision',
    'recall': 'Recall',
    'sensitivity': 'Recall',
    'average_precision': 'Average Precision',
    'average_recall': 'Average Recall',
    'ap': 'Average Precision',
    'ar': 'Average Recall',
    'f1': 'F1',
    'fpr': 'FPR',
    'fnr': 'FNR'
}


def convert_classification_scorers(scorer: t.Union[Metric, str, t.Callable]):
    classification_scores = get_default_classification_scorers()
    if isinstance(scorer, str):
        scorer_name = _func_naming_dict.get(scorer.lower().replace(' ', '_'), scorer).lower()
        if scorer_name in classification_scores:
            return classification_scores[scorer_name]
    if isinstance(scorer, (_BaseScorer, str)):
        scorer = DeepcheckScorer(scorer, '')
        needs_proba = isinstance(scorer, _ProbaScorer)
        return CustomScorer(scorer.run_on_pred, needs_proba=needs_proba)
    elif callable(scorer):
        return CustomScorer(scorer)
    return None


def convert_detection_scorers(scorer: t.Union[Metric, str, t.Callable]):
    if isinstance(scorer, str):
        scorer_name = _func_naming_dict.get(scorer.lower().replace(' ', '_'), scorer).lower()
        detection_scores = get_default_object_detection_scorers()
        if scorer_name in detection_scores:
            return detection_scores[scorer_name]
        if scorer_name in AVAILABLE_EVALUTING_FUNCTIONS:
            return ObjectDetectionTpFpFn(evaluting_function=scorer_name)
    return None


def get_scorers_list(
        dataset: VisionData,
        alternative_scorers: t.Union[t.Dict[str, t.Union[Metric, t.Callable, str]], t.List[t.Any]] = None,
) -> t.Dict[str, Metric]:
    """Get scorers list according to model object and label column.

    Parameters
    ----------
    dataset : VisionData
        Dataset object
    alternative_scorers : t.Union[t.Dict[str, t.Union[Metric, t.Callable, str]], t.List[str]], default: None
        Alternative scorers dictionary (or a list)
    Returns
    -------
    t.Dict[str, Metric]
        Scorers list
    """
    task_type = dataset.task_type

    if alternative_scorers:
        # For alternative scorers we create a copy since in suites we are running in parallel, so we can't use the same
        # instance for several checks.
        if isinstance(alternative_scorers, list):
            alternative_scorers = {(name if isinstance(name, str) else type(name).__name__):
                                   name for name in alternative_scorers}
        scorers = {}
        for name, met in alternative_scorers.items():
            # Validate that each alternative scorer is a correct type
            if isinstance(met, Metric):
                met.reset()
                scorers[name] = copy(met)
            elif isinstance(met, str) or callable(met):
                if task_type == TaskType.OBJECT_DETECTION:
                    met = convert_detection_scorers(met)
                else:
                    met = convert_classification_scorers(met)
                if met is None:
                    raise DeepchecksNotSupportedError(
                        f'Unsupported metric: {name} of type {type(met).__name__} was given.')
            raise DeepchecksValueError(
                f'Excepted metric type one of [ignite.Metric, str, callable], was {type(met).__name__}.')

        return scorers
    elif task_type == TaskType.CLASSIFICATION:
        scorers = get_default_classification_scorers()
    elif task_type == TaskType.OBJECT_DETECTION:
        scorers = get_default_object_detection_scorers()
    else:
        raise DeepchecksNotSupportedError(f'No scorers match task_type {task_type}')

    return scorers


def calculate_metrics(
    metrics: t.Dict[str, Metric],
    dataset: VisionData,
    model: torch.nn.Module,
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
    device : Union[str, torch.device, None]

    Returns
    -------
    t.Dict[str, float]
        Dictionary of metrics with the metric name as key and the metric value as value
    """

    def process_function(_, batch):
        return dataset.infer_on_batch(batch, model, device), dataset.batch_to_labels(batch)

    engine = Engine(process_function)

    for name, metric in metrics.items():
        metric.reset()
        metric.attach(engine, name)

    state = engine.run(dataset.data_loader)
    return state.metrics


def _validate_metric_type(metric_name: str, score: t.Any) -> bool:
    """Raise error if metric has incorrect type, or return true."""
    if not isinstance(score, (torch.Tensor, list, np.ndarray)):
        raise DeepchecksValueError(f'The metric {metric_name} returned a '
                                   f'{type(score)} instead of an array/tensor')
    return True


def metric_results_to_df(results: dict, dataset: VisionData) -> pd.DataFrame:
    """Get dict of metric name to tensor of classes scores, and convert it to dataframe."""
    # The data might contain fewer classes than the model was trained on. filtering out any class id which is not
    # presented in the data.
    data_classes = dataset.classes_indices.keys()

    per_class_result = [
        [metric, class_id, dataset.label_id_to_name(class_id),
         class_score.item() if isinstance(class_score, torch.Tensor) else class_score]
        for metric, score in results.items()
        if _validate_metric_type(metric, score)
        # scorer returns results as array, containing result per class
        for class_id, class_score in enumerate(score)
        if not np.isnan(class_score) and class_id in data_classes
    ]

    return pd.DataFrame(per_class_result, columns=['Metric',
                                                   'Class',
                                                   'Class Name',
                                                   'Value']).sort_values(by=['Metric', 'Class'])


def filter_classes_for_display(metrics_df: pd.DataFrame,
                               metric_to_show_by: str,
                               n_to_show: int,
                               show_only: str,
                               column_to_filter_by: str = 'Dataset',
                               column_filter_value: str = None) -> list:
    """Filter the metrics dataframe for display purposes.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing the metrics.
    n_to_show : int
        Number of classes to show in the report.
    show_only : str
        Specify which classes to show in the report. Can be one of the following:
        - 'largest': Show the largest classes.
        - 'smallest': Show the smallest classes.
        - 'random': Show random classes.
        - 'best': Show the classes with the highest score.
        - 'worst': Show the classes with the lowest score.
    metric_to_show_by : str
        Specify the metric to sort the results by. Relevant only when show_only is 'best' or 'worst'.
    column_to_filter_by : str , default: 'Dataset'
        Specify the name of the column to filter by.
    column_filter_value : str , default: None
        Specify the value of the column to filter by, if None will be set to test dataset name.

    Returns
    -------
    list
        List of classes to show in the report.
    """
    # working on the test dataset on default
    if column_filter_value is None:
        column_filter_value = DatasetKind.TEST.value

    tests_metrics_df = metrics_df[(metrics_df[column_to_filter_by] == column_filter_value) &
                                  (metrics_df['Metric'] == metric_to_show_by)]
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
