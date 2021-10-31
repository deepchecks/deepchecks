from typing import Callable, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from mlchecks import Dataset, CheckResult
from mlchecks.metric_utils import validate_scorer, task_type_check, DEFAULT_SINGLE_METRIC, DEFAULT_METRICS_DICT
from mlchecks.string_utils import format_number
from mlchecks.utils import MLChecksValueError
import matplotlib.pyplot as plt

__all__ = ['segment_performance']


def column_values(dataset: Dataset, numeric_bins: int, column: str):
    if column in dataset.cat_features():
        return dataset.data[column].unique()
    else:
        min_val = min(dataset.data[column])
        max_val = max(dataset.data[column])
        bins = np.linspace(0, 1, numeric_bins + 1) * (max_val - min_val) + min_val
        return list(zip(bins[:-1], bins[1:]))


def filter_dataframe(data: pd.DataFrame, column: str, value, is_last):
    if isinstance(value, Tuple):
        # Only in last range include equals for the end of the range
        if is_last:
            return data.loc[(data[column] >= value[0]) & (data[column] <= value[1])]
        else:
            return data.loc[(data[column] >= value[0]) & (data[column] < value[1])]
    else:
        return data.loc[data[column] == value]


def create_labels(values):
    if isinstance(values[0], Tuple):
        start_values, last_value = values[:-1], values[-1]
        labels = [f'[{format_number(start)} - {format_number(end)})' for start, end in start_values]
        # Last range is closed in the end
        labels.append(f'[{format_number(last_value[0])} - {format_number(last_value[1])}]')
        return labels
    else:
        return [str(v) for v in values]


def segment_performance(dataset: Dataset, model, metric: Union[str, Callable] = None,
                        feature_1: str = None, feature_2: str = None, numeric_bins: int = 10):
    self = segment_performance
    # Validations
    if feature_1 is None or feature_2 is None:
        raise MLChecksValueError('Must define both `feature_1` and `feature_2`')

    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)
    dataset.validate_model(model)

    if metric is not None:
        scorer = validate_scorer(metric, model, dataset)
        metric_name = metric if isinstance(metric, str) else 'User metric'
    else:
        model_type = task_type_check(model, dataset)
        metric_name = DEFAULT_SINGLE_METRIC[model_type]
        scorer = DEFAULT_METRICS_DICT[model_type][metric_name]

    feature_1_values = column_values(dataset, numeric_bins, feature_1)
    feature_2_values = column_values(dataset, numeric_bins, feature_2)

    scores = np.empty((len(feature_1_values), len(feature_2_values)), dtype=float)
    for i, feature_1_value in enumerate(feature_1_values):
        data = dataset.data
        feature_1_last_iteration = feature_1_value == feature_1_values[-1]
        feature_1_filtered = filter_dataframe(data, feature_1, feature_1_value, feature_1_last_iteration)

        for j, feature_2_value in enumerate(feature_2_values):
            feature_2_last_iteration = feature_2_value == feature_2_values[-1]
            feature_2_filtered = filter_dataframe(feature_1_filtered, feature_2, feature_2_value,
                                                  feature_2_last_iteration)

            # Run on filtered data and save to matrix
            if feature_2_filtered.empty:
                score = np.NaN
            else:
                score = scorer(model, feature_2_filtered[dataset.features()], feature_2_filtered[dataset.label_name()])
            scores[i, j] = score

    def display():
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        im = ax.imshow(np.array(scores, dtype=float))

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Score', rotation=-90, va="bottom")

        x = feature_2_values
        y = feature_1_values

        # Set ticks with labels
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(y)))
        ax.set_xticklabels(create_labels(x))
        ax.set_yticklabels(create_labels(y))

        plt.xlabel(feature_2)
        plt.ylabel(feature_1)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(y)):
            for j in range(len(x)):
                if not np.isnan(scores[i, j]):
                    ax.text(j, i, format_number(scores[i, j]), ha="center", va="center", color="w")

        ax.set_title(f'{metric_name} by features {feature_1}/{feature_2}')
        # fig.tight_layout()

    return CheckResult(scores, check=self, display=display)
