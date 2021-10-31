from typing import Callable, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from mlchecks import Dataset, CheckResult
from mlchecks.checks.performance.partition import partition_feature_to_bins, MLChecksFilter
from mlchecks.metric_utils import validate_scorer, task_type_check, DEFAULT_SINGLE_METRIC, DEFAULT_METRICS_DICT
from mlchecks.string_utils import format_number
from mlchecks.utils import MLChecksValueError
import matplotlib.pyplot as plt

__all__ = ['segment_performance']


# def filter_dataframe(data: pd.DataFrame, column: str, value):
#     if isinstance(value, Tuple):
#         return data.loc[(data[column] >= value[0]) & (data[column] < value[1])]
#     else:
#         if isinstance(value, MLChecksFilter):
# #             return data.loc[data[column] == value]
#
#
# def create_labels(values):
#     if isinstance(values[0], Tuple):
#         start_values, last_value = values[:-1], values[-1]
#         labels = [f'[{format_number(start)} - {format_number(end)})' for start, end in start_values]
#         # Last range is closed in the end
#         labels.append(f'[{format_number(last_value[0])} - {format_number(last_value[1])}]')
#         return labels
#     else:
#         return [str(v) for v in values]


def segment_performance(dataset: Dataset, model, metric: Union[str, Callable] = None,
                        feature_1: str = None, feature_2: str = None, num_segments: int = 10):
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

    feature_1_filters = partition_feature_to_bins(dataset, feature_1, max_segments=num_segments)
    feature_2_filters = partition_feature_to_bins(dataset, feature_2, max_segments=num_segments)

    scores = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=float)
    for i, feature_1_filter in enumerate(feature_1_filters):
        data = dataset.data
        feature_1_df = feature_1_filter.filter(data)
        for j, feature_2_filter in enumerate(feature_2_filters):
            feature_2_df = feature_2_filter.filter(feature_1_df)

            # Run on filtered data and save to matrix
            if feature_2_df.empty:
                score = np.NaN
            else:
                score = scorer(model, feature_2_df[dataset.features()], feature_2_df[dataset.label_name()])
            scores[i, j] = score

    def display():
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        im = ax.imshow(np.array(scores, dtype=float))

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Score', rotation=-90, va="bottom")

        x = [v.label for v in feature_2_filters]
        y = [v.label for v in feature_1_filters]

        # Set ticks with labels
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(y)))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)

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
