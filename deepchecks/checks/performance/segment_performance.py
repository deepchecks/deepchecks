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
from typing import Callable, Union, Optional, List, cast, Tuple

import numpy as np
import plotly.figure_factory as ff

from deepchecks.base.check_context import CheckRunContext
from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.performance.partition import partition_column
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError, DatasetValidationError


__all__ = ['SegmentPerformance']


class SegmentPerformance(SingleDatasetBaseCheck):
    """Display performance score segmented by 2 top (or given) features in a heatmap.

    Parameters
    ----------
    feature_1 : Optional[Hashable] , default: None
        feature to segment by on y-axis.
    feature_2 : Optional[Hashable] , default: None
        feature to segment by on x-axis.
    alternative_scorer : Tuple[str, Union[str, Callable]] , default: None
        Score to show, either function or sklearn scorer name.
        If is not given a default scorer (per the model type) will be used.
    max_segments : int , default: 10
        maximal number of segments to split the values into.
    """

    feature_1: Optional[Hashable]
    feature_2: Optional[Hashable]
    scorer: Union[str, Callable, None]
    max_segments: int

    def __init__(
        self,
        feature_1: Optional[Hashable] = None,
        feature_2: Optional[Hashable] = None,
        alternative_scorer: Tuple[str, Union[str, Callable]] = None,
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
        self.user_scorer = dict([alternative_scorer]) if alternative_scorer else None

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        model = context.model
        features = context.features
        features_importance = context.features_importance
        scorer = context.get_single_scorer(self.user_scorer)

        if len(features) < 2:
            raise DatasetValidationError('Dataset must have at least 2 features')

        if self.feature_1 is None and self.feature_2 is None:
            if features_importance is None:
                self.feature_1, self.feature_2, *_ = features
            else:
                features_importance.sort_values(ascending=False, inplace=True)
                self.feature_1, self.feature_2, *_ = cast(List[Hashable], list(features_importance.keys()))

        elif self.feature_1 is None or self.feature_2 is None:
            raise DeepchecksValueError('Must define both "feature_1" and "feature_2" or none of them')

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
                    score = scorer(model,
                                   Dataset(feature_2_df, features=dataset.features,
                                           label=dataset.label_name, cat_features=dataset.cat_features))
                scores[i, j] = score
                counts[i, j] = len(feature_2_df)

        x = [v.label for v in feature_2_filters]
        y = [v.label for v in feature_1_filters]

        scores_text = [[0]*scores.shape[1] for _ in range(scores.shape[0])]

        for i in range(len(y)):
            for j in range(len(x)):
                score = scores[i, j]
                if not np.isnan(score):
                    scores_text[i][j] = f'{format_number(score)}\n({counts[i, j]})'
                else:
                    scores_text[i][j] = f'{score}\n({counts[i, j]})'

        fig = ff.create_annotated_heatmap(scores, annotation_text=scores_text,
                                          x=x, y=y, colorscale='rdylgn', font_colors=['black', 'black'])
        fig.update_layout(title=f'{scorer.name} (count) by features {self.feature_1}/{self.feature_2}',
                          width=800, height=800)
        fig.update_xaxes(title=self.feature_2)
        fig.update_yaxes(title=self.feature_1, autorange='reversed')
        fig['data'][0]['showscale'] = True
        fig['layout']['xaxis']['side'] = 'bottom'

        value = {'scores': scores, 'counts': counts, 'feature_1': self.feature_1, 'feature_2': self.feature_2}
        return CheckResult(value, display=fig)
