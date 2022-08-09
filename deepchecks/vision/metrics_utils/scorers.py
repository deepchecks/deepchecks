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
"""Module for defining scorers for the vision module."""
import typing as t
from copy import copy
from numbers import Number

import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine
from ignite.metrics import Metric, Precision, Recall
from sklearn.metrics._scorer import _ProbaScorer

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.utils.metrics import get_scorer_name
from deepchecks.vision.metrics_utils import (CustomClassificationScorer, ObjectDetectionAveragePrecision,
                                             ObjectDetectionTpFpFn)
from deepchecks.vision.vision_data import TaskType, VisionData

__all__ = [
    'get_scorers_dict',
    'calculate_metrics',
    'metric_results_to_df',
    'filter_classes_for_display',
]


def get_default_classification_scorers():
    return {
        'Precision': classification_dict['precision_per_class'](),
        'Recall':  classification_dict['recall_per_class']()
    }


def get_default_object_detection_scorers() -> t.Dict[str, Metric]:
    return {
        'Average Precision': detection_dict['average_precision_per_class'](),
        'Average Recall': detection_dict['average_recall_per_class']()
    }


classification_dict = {
    'precision_per_class': Precision,
    'recall_per_class': Recall,
}

detection_dict = {
    'precision_per_class': lambda: ObjectDetectionTpFpFn(evaluating_function='precision', averaging_method='per_class'),
    'precision': lambda: ObjectDetectionTpFpFn(evaluating_function='precision', averaging_method='binary'),
    'precision_macro': lambda: ObjectDetectionTpFpFn(evaluating_function='precision', averaging_method='macro'),
    'precision_micro': lambda: ObjectDetectionTpFpFn(evaluating_function='precision', averaging_method='micro'),
    'precision_weighted': lambda: ObjectDetectionTpFpFn(evaluating_function='precision', averaging_method='weighted'),
    'recall_per_class': lambda: ObjectDetectionTpFpFn(evaluating_function='recall', averaging_method='per_class'),
    'recall': lambda: ObjectDetectionTpFpFn(evaluating_function='recall', averaging_method='binary'),
    'recall_macro': lambda: ObjectDetectionTpFpFn(evaluating_function='recall', averaging_method='macro'),
    'recall_micro': lambda: ObjectDetectionTpFpFn(evaluating_function='recall', averaging_method='micro'),
    'recall_weighted': lambda: ObjectDetectionTpFpFn(evaluating_function='recall', averaging_method='weighted'),
    'f1_per_class': lambda: ObjectDetectionTpFpFn(evaluating_function='f1', averaging_method='per_class'),
    'f1': lambda: ObjectDetectionTpFpFn(evaluating_function='f1', averaging_method='binary'),
    'f1_macro': lambda: ObjectDetectionTpFpFn(evaluating_function='f1', averaging_method='macro'),
    'f1_micro': lambda: ObjectDetectionTpFpFn(evaluating_function='f1', averaging_method='micro'),
    'f1_weighted': lambda: ObjectDetectionTpFpFn(evaluating_function='f1', averaging_method='weighted'),
    'fpr_per_class': lambda: ObjectDetectionTpFpFn(evaluating_function='fpr', averaging_method='per_class'),
    'fpr': lambda: ObjectDetectionTpFpFn(evaluating_function='fpr', averaging_method='binary'),
    'fpr_macro': lambda: ObjectDetectionTpFpFn(evaluating_function='fpr', averaging_method='macro'),
    'fpr_micro': lambda: ObjectDetectionTpFpFn(evaluating_function='fpr', averaging_method='micro'),
    'fpr_weighted': lambda: ObjectDetectionTpFpFn(evaluating_function='fpr', averaging_method='weighted'),
    'fnr_per_class': lambda: ObjectDetectionTpFpFn(evaluating_function='fnr', averaging_method='per_class'),
    'fnr': lambda: ObjectDetectionTpFpFn(evaluating_function='fnr', averaging_method='binary'),
    'fnr_macro': lambda: ObjectDetectionTpFpFn(evaluating_function='fnr', averaging_method='macro'),
    'fnr_micro': lambda: ObjectDetectionTpFpFn(evaluating_function='fnr', averaging_method='micro'),
    'fnr_weighted': lambda: ObjectDetectionTpFpFn(evaluating_function='fnr', averaging_method='weighted'),
    'average_precision_per_class': lambda: ObjectDetectionAveragePrecision(return_option='ap'),
    'average_recall_per_class': lambda: ObjectDetectionAveragePrecision(return_option='ar')
}


def get_scorers_dict(
        dataset: VisionData,
        alternative_scorers: t.Union[t.Dict[str, t.Union[Metric, str]], t.List[t.Union[Metric, str]]] = None,
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
            alternative_scorers = {get_scorer_name(scorer): scorer for scorer in alternative_scorers}
        scorers = {}
        for name, metric in alternative_scorers.items():
            # Validate that each alternative scorer is a correct type
            if isinstance(metric, Metric):
                metric.reset()
                scorers[name] = copy(metric)
            elif isinstance(metric, str):
                metric_name = metric.lower().replace(' ', '_').replace('sensitivity', 'recall')
                if task_type == TaskType.OBJECT_DETECTION and metric_name in detection_dict:
                    converted_met = detection_dict[metric_name]()
                elif task_type == TaskType.CLASSIFICATION:
                    if metric_name in classification_dict:
                        converted_met = classification_dict[metric_name]()
                    else:
                        scorer = DeepcheckScorer(metric_name)
                        needs_proba = isinstance(scorer, _ProbaScorer)
                        converted_met = CustomClassificationScorer(scorer.run_on_pred, needs_proba=needs_proba)
                else:
                    raise DeepchecksNotSupportedError(
                        f'Unsupported metric: {name} of type {type(metric).__name__} was given.')
                scorers[name] = converted_met
            else:
                raise DeepchecksValueError(
                    f'Excepted metric type one of [ignite.Metric, str], was {type(metric).__name__}.')
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


def metric_results_to_df(results: dict, dataset: VisionData) -> pd.DataFrame:
    """Get dict of metric name to tensor of classes scores, and convert it to dataframe."""
    data_classes = dataset.classes_indices.keys()
    result_list = []

    for metric, scores in results.items():
        if isinstance(scores, Number):
            result_list.append([metric, pd.NA, pd.NA, scores])
        elif len(scores) == 1:
            score = scores[0].item() if isinstance(scores[0], torch.Tensor) else scores[0]
            result_list.append([metric, pd.NA, pd.NA, score])
        elif isinstance(scores, (torch.Tensor, list, np.ndarray)):
            for class_id, class_score in enumerate(scores):
                # The data might contain fewer classes than the model was trained on. filtering out
                # any class id which is not presented in the data.
                if np.isnan(class_score) or class_id not in data_classes:
                    continue
                score = class_score.item() if isinstance(class_score, torch.Tensor) else class_score
                result_list.append([metric, class_id, dataset.label_id_to_name(class_id), score])
        else:
            raise DeepchecksValueError(f'The metric {metric} returned a '
                                       f'{type(scores)} instead of an array/tensor')

    return pd.DataFrame(result_list, columns=['Metric', 'Class', 'Class Name', 'Value'])


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
