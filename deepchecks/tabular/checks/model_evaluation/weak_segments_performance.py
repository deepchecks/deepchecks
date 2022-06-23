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
"""Module of weak segments performance check."""
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import sklearn
from category_encoders import TargetEncoder
from packaging import version
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from deepchecks import ConditionCategory, ConditionResult, Dataset
from deepchecks.core import CheckResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.performance.partition import convert_tree_leaves_into_filters
from deepchecks.utils.single_sample_metrics import calculate_per_sample_loss
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['WeakSegmentsPerformance']


class WeakSegmentsPerformance(SingleDatasetCheck):
    """Search for 2 feature based segments with the lowest performance score.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_features : int , default: 5
        Amount of features to use for segment search. Select top columns based on feature importance to error model.
    segment_minimum_size_ratio: float , default: 0.01
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    alternative_scorer : Tuple[str, Union[str, Callable]] , default: None
        Score to show, either function or sklearn scorer name.
        If is not given a default scorer (per the model type) will be used.
    loss_per_sample: Union[np.array, pd.Series, None], default: None
        Loss per sample used to detect relevant weak segments. If set to none uses log loss for classification
        and mean square error for regression.
    n_samples : int , default: 10_000
        number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_top_features: int = 5,
            segment_minimum_size_ratio: float = 0.01,
            alternative_scorer: Dict[str, Callable] = None,
            loss_per_sample: Union[np.array, pd.Series, None] = None,
            classes_index_order: Union[np.array, pd.Series, None] = None,
            n_samples: int = 10_000,
            n_to_show: int = 3,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_features = n_top_features
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state
        self.loss_per_sample = loss_per_sample
        self.classes_index_order = classes_index_order
        self.user_scorer = alternative_scorer if alternative_scorer else None

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset.assert_features()
        dataset = dataset.sample(self.n_samples, random_state=self.random_state, drop_na_label=True)
        predictions = context.model.predict(dataset.features_columns)
        y_proba = context.model.predict_proba(dataset.features_columns) if \
            context.task_type in [TaskType.MULTICLASS, TaskType.BINARY] else None
        dataset = dataset.select(self.columns, self.ignore_columns)

        if self.loss_per_sample is not None:
            loss_per_sample = self.loss_per_sample[list(dataset.data.index)]
        else:
            loss_per_sample = calculate_per_sample_loss(context.model, context.task_type, dataset,
                                                        self.classes_index_order)

        if len(dataset.cat_features) > 0:
            t_encoder = TargetEncoder(cols=dataset.cat_features)
            df_encoded = t_encoder.fit_transform(dataset.features_columns, dataset.label_col)
        else:
            df_encoded = dataset.features_columns
        df_encoded = df_encoded.fillna(df_encoded.mean(axis=0))
        encoded_dataset = Dataset(df_encoded, cat_features=[], label=dataset.label_col)
        dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=predictions, y_proba_test=y_proba)
        feature_importance = context.features_importance.sort_values(ascending=False)

        scorer = context.get_single_scorer(self.user_scorer)
        weak_segments = self._weak_segments_search(dummy_model, encoded_dataset, feature_importance,
                                                   loss_per_sample, scorer)
        avg_score = round(scorer(dummy_model, encoded_dataset), 3)

        return CheckResult({'segments': weak_segments, 'avg_score': avg_score, 'scorer_name': scorer.name},
                           display=[weak_segments.iloc[:self.n_to_show, :]])

    def _weak_segments_search(self, dummy_model, encoded_dataset, feature_rank_for_search, loss_per_sample, scorer):
        """Search for weak segments based on scorer."""
        weak_segments = pd.DataFrame(
            columns=[f'segment_score_{scorer.name}', 'feature1', 'feature1_range', 'feature2', 'feature2_range'])
        for i in range(min(len(feature_rank_for_search.keys()), self.n_top_features)):
            for j in range(i + 1, min(len(feature_rank_for_search.keys()), self.n_top_features)):
                feature1, feature2 = feature_rank_for_search.keys()[[i, j]]
                weak_segment_score, weak_segment_filter = self._find_weak_segment(dummy_model, encoded_dataset,
                                                                                  [feature1, feature2], scorer,
                                                                                  loss_per_sample)
                weak_segments.loc[len(weak_segments)] = [round(weak_segment_score, 3), feature1,
                                                         np.around(weak_segment_filter.filters[feature1], 3), feature2,
                                                         np.around(weak_segment_filter.filters[feature2], 3)]
        return weak_segments.sort_values(f'segment_score_{scorer.name}')

    def _find_weak_segment(self, dummy_model, dataset, features_for_segment, scorer, loss_per_sample):
        """Find weak segment based on scorer for specified features."""
        if version.parse(sklearn.__version__) < version.parse('1.0.0'):
            criterion = ['mse', 'mae']
        else:
            criterion = ['squared_error', 'absolute_error']
        search_space = {
            'max_depth': [5],
            'min_weight_fraction_leaf': [self.segment_minimum_size_ratio],
            'min_samples_leaf': [10],
            'criterion': criterion
        }

        def get_worst_leaf_filter(tree):
            leaves_filters = convert_tree_leaves_into_filters(tree, features_for_segment)
            min_score, min_score_leaf_filter = np.inf, None
            for leaf_filter in leaves_filters:
                leaf_data = leaf_filter.filter(dataset.data)
                leaf_score = scorer(dummy_model,
                                    Dataset(leaf_data.iloc[:, :-1], cat_features=[], label=leaf_data.iloc[:, -1]))
                if leaf_score < min_score:
                    min_score, min_score_leaf_filter = leaf_score, leaf_filter
            return min_score, min_score_leaf_filter

        def neg_worst_segment_score(clf: DecisionTreeRegressor, x, y) -> float:  # pylint: disable=unused-argument
            return -get_worst_leaf_filter(clf.tree_)[0]

        random_searcher = GridSearchCV(DecisionTreeRegressor(), scoring=neg_worst_segment_score,
                                       param_grid=search_space, n_jobs=-1, cv=3)
        random_searcher.fit(dataset.features_columns[features_for_segment], loss_per_sample)
        segment_score, segment_filter = get_worst_leaf_filter(random_searcher.best_estimator_.tree_)

        if features_for_segment[0] not in segment_filter.filters.keys():
            segment_filter.filters[features_for_segment[0]] = [np.NINF, np.inf]
        if features_for_segment[1] not in segment_filter.filters.keys():
            segment_filter.filters[features_for_segment[1]] = [np.NINF, np.inf]
        return segment_score, segment_filter

    def add_condition_segments_performance_relative_difference_greater_than(self, max_ratio_change: float = 0.20):
        """Add condition - check that the score of the weakest segment is at least (1 - ratio) * average score.

        Parameters
        ----------
        max_ratio_change : float , default: 0.20
            maximal ratio of change between the average score and the score of the weakest segment.
        """

        def condition(result: Dict) -> ConditionResult:
            weakest_segment_score = result['segments'].iloc[0, 0]
            msg = f'Weakest segment score of {format_number(weakest_segment_score, 3)} in comparison to average ' \
                  f'score of {format_number(result["avg_score"], 3)} based on scorer {result["scorer_name"]}.'
            if weakest_segment_score < (1 - max_ratio_change) * result['avg_score']:
                return ConditionResult(ConditionCategory.WARN, msg)
            else:
                return ConditionResult(ConditionCategory.PASS, msg)

        return self.add_condition(f'The performance of weakest segment is greater than '
                                  f'{format_percent(1 - max_ratio_change)} of average model performance.', condition)
