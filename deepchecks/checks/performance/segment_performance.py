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
"""Module of segment performance check."""
from typing import Callable, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.checks.performance.partition import partition_column
from deepchecks.utils.metrics import validate_scorer, task_type_check, DEFAULT_SINGLE_METRIC, DEFAULT_METRICS_DICT
from deepchecks.utils.strings import format_number
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.validation import validate_model
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


__all__ = ['SegmentPerformance']


class SegmentPerformance(SingleDatasetBaseCheck):
    """Display performance metric segmented by 2 top (or given) features in a heatmap.

    Args:
        feature_1 (Hashable): feature to segment by on y-axis.
        feature_2 (Hashable): feature to segment by on x-axis.
        metric (Union[str, Callable]): Metric to show, either function or sklearn scorer name. If no metric is given
            a default metric (per the model type) will be used.
        max_segments (int): maximal number of segments to split the a values into.
    """

    feature_1: Optional[Hashable]
    feature_2: Optional[Hashable]
    metric: Union[str, Callable, None]
    max_segments: int

    def __init__(
        self,
        feature_1: Optional[Hashable] = None,
        feature_2: Optional[Hashable] = None,
        metric: Union[str, Callable] = None,
        max_segments: int = 10
    ):
        super().__init__()

        # if they're both none it's ok
        if feature_1 and feature_1 == feature_2:
            raise DeepchecksValueError('"feature_1" must be different than "feature_2"')
        self.feature_1 = feature_1
        self.feature_2 = feature_2

        if not isinstance(max_segments, int) or max_segments < 0:
            raise DeepchecksValueError('"num_segments" must be positive integer')
        self.max_segments = max_segments
        self.metric = metric

    def run(self, dataset, model) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        """
        # Validations
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        validate_model(dataset, model)

        if self.feature_1 is None or self.feature_2 is None:
            # only one none is not ok
            if self.feature_1 is None and self.feature_2 is None:
                feature_importance = calculate_feature_importance(dataset=dataset, model=model)
                if len(feature_importance) < 2:
                    raise DeepchecksValueError('Must have at least 2 features')
                feature_importance.sort_values(ascending=False, inplace=True)
                self.feature_1, self.feature_2 = feature_importance.keys()[0], feature_importance.keys()[1]
            else:
                raise DeepchecksValueError('Must define both "feature_1" and "feature_2" or none of them')

        if self.metric is not None:
            scorer = validate_scorer(self.metric, model, dataset)
            metric_name = self.metric if isinstance(self.metric, str) else 'User metric'
        else:
            model_type = task_type_check(model, dataset)
            metric_name = DEFAULT_SINGLE_METRIC[model_type]
            scorer = DEFAULT_METRICS_DICT[model_type][metric_name]

        feature_1_filters = partition_column(dataset, self.feature_1, max_segments=self.max_segments)
        feature_2_filters = partition_column(dataset, self.feature_2, max_segments=self.max_segments)

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
                    score = scorer(model, feature_2_df[dataset.features], feature_2_df[dataset.label_name])
                scores[i, j] = score
                counts[i, j] = len(feature_2_df)

        def display(feat1=self.feature_1, feat2=self.feature_2):
            ax: Axes
            _, ax = plt.subplots(1, 1, figsize=(10, 7))
            im = ax.imshow(np.array(scores, dtype=float), cmap='RdYlGn')

            # Create colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(f'{metric_name}', rotation=-90, va='bottom')

            x = [v.label for v in feature_2_filters]
            y = [v.label for v in feature_1_filters]

            # Set ticks with labels
            ax.set_xticks(np.arange(len(x)), minor=False)
            ax.set_yticks(np.arange(len(y)), minor=False)
            ax.set_xticklabels(x, minor=False)
            ax.set_yticklabels(y, minor=False)

            plt.xlabel(feat2)
            plt.ylabel(feat1)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            # Loop over data dimensions and create text annotations.
            q1, q2 = np.nanquantile(scores.flatten(), [0.1, 0.9])
            for i in range(len(y)):
                for j in range(len(x)):
                    score = scores[i, j]
                    if not np.isnan(score):
                        # The largest and smallest scores have dark background, so give them white text color
                        color = 'black' if q1 < score < q2 else 'white'
                        text = f'{format_number(score)}\n({counts[i, j]})'
                        ax.text(j, i, text, ha='center', va='center', color=color)

            ax.set_title(f'{metric_name} (count) by features {feat1}/{feat2}')

        value = {'scores': scores, 'counts': counts, 'feature_1': self.feature_1,'feature_2': self.feature_2}
        return CheckResult(value, display=display)
