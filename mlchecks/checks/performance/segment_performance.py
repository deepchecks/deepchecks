"""Module of segment performance check."""
from typing import Callable, Union
import numpy as np
from matplotlib.axes import Axes
from mlchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from mlchecks.checks.performance.partition import partition_column
from mlchecks.metric_utils import validate_scorer, task_type_check, DEFAULT_SINGLE_METRIC, DEFAULT_METRICS_DICT
from mlchecks.string_utils import format_number
from mlchecks.utils import MLChecksValueError
import matplotlib.pyplot as plt

__all__ = ['segment_performance']


def segment_performance(dataset: Dataset, model, metric: Union[str, Callable] = None,
                        feature_1: str = None, feature_2: str = None, max_segments: int = 10):
    """Display performance metric segmented by 2 given features in a heatmap.

    Args:
        dataset (Dataset): a Dataset object.
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        metric (Union[str, Callable]): Metric to show, either function or sklearn scorer name.
        feature_1 (str): feature to segment by on y-axis.
        feature_2 (str): feature to segment by on x-axis.
        max_segments (int): maximal number of segments to split the a values into.
    """
    self = segment_performance
    # Validations
    if feature_1 is None or feature_2 is None:
        raise MLChecksValueError('Must define both "feature_1" and "feature_2"')
    # if feature_1 == feature_2:
    #     raise MLChecksValueError('"feature_1" must be different than "feature_2"')
    if not isinstance(max_segments, int) or max_segments < 0:
        raise MLChecksValueError('"num_segments" must be positive integer')

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

    feature_1_filters = partition_column(dataset, feature_1, max_segments=max_segments)
    feature_2_filters = partition_column(dataset, feature_2, max_segments=max_segments)

    scores = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=float)
    counts = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=int)

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
            counts[i, j] = len(feature_2_df)

    def display():
        ax: Axes
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Tahoma']
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'

        _, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(np.array(scores, dtype=float), cmap='RdYlGn')

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Score', rotation=-90, va='bottom')

        x = [v.label for v in feature_2_filters]
        y = [v.label for v in feature_1_filters]

        # Set ticks with labels
        ax.set_xticks(np.arange(len(x)), minor=False)
        ax.set_yticks(np.arange(len(y)), minor=False)
        ax.set_xticklabels(x, minor=False)
        ax.set_yticklabels(y, minor=False)

        plt.xlabel(feature_2)
        plt.ylabel(feature_1)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        q1, q2 = np.nanquantile(scores.flatten(), [0.1, 0.9])
        for i in range(len(y)):
            for j in range(len(x)):
                score = scores[i, j]
                if not np.isnan(score):
                    color = 'black' if q1 < score < q2 else 'white'
                    text = f'{format_number(score)}\n({counts[i, j]})'
                    ax.text(j, i, text, ha='center', va='center', color=color)

        ax.set_title(f'{metric_name} (count) by features {feature_1}/{feature_2}')

    return CheckResult(scores, check=self, display=display)


class SegmentPerformance(SingleDatasetBaseCheck):
    """Display performance metric segmented by 2 given features in a heatmap.

    Params:
        metric (Union[str, Callable]): Metric to show, either function or sklearn scorer name.
        feature_1 (str): feature to segment by on y-axis.
        feature_2 (str): feature to segment by on x-axis.
        max_segments (int): maximal number of segments to split the a values into.
    """

    def run(self, dataset, model=None) -> CheckResult:
        """Run 'segment_performance' check.

        Args:
            dataset (Dataset): a Dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        """
        return segment_performance(dataset, model, **self.params)
