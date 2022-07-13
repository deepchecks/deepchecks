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
from collections import defaultdict
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from category_encoders import TargetEncoder
from packaging import version
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError
from deepchecks.tabular import Context, Dataset, SingleDatasetCheck
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.metric_utils.scorers import DeepcheckScorer
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.dataframes import default_fill_na_per_column_type
from deepchecks.utils.performance.partition import (convert_tree_leaves_into_filters,
                                                    partition_numeric_feature_around_segment)
from deepchecks.utils.single_sample_metrics import calculate_per_sample_loss
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['WeakSegmentsPerformance']


class WeakSegmentsPerformance(SingleDatasetCheck):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.
    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_features : int , default: 5
        Number of features to use for segment search. Top columns are selected based on feature importance.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    alternative_scorer : Tuple[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    loss_per_sample: Union[np.array, pd.Series, None], default: None
        Loss per sample used to detect relevant weak segments. If pd.Series the indexes should be similar to those in
        the dataset object provide, if np.array the order should be based on the index order of the dataset object and
        if None the check calculates loss per sample by via log loss for classification and MSE for regression.
    n_samples : int , default: 10_000
        number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_top_features: int = 5,
            segment_minimum_size_ratio: float = 0.05,
            alternative_scorer: Dict[str, Callable] = None,
            loss_per_sample: Union[np.array, pd.Series, None] = None,
            classes_index_order: Union[np.array, pd.Series, None] = None,
            n_samples: int = 10_000,
            categorical_aggregation_threshold: float = 0.05,
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
        self.categorical_aggregation_threshold = categorical_aggregation_threshold

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset.assert_features()
        dataset = dataset.sample(self.n_samples, random_state=self.random_state, drop_na_label=True)
        predictions = context.model.predict(dataset.features_columns)
        y_proba = context.model.predict_proba(dataset.features_columns) if \
            context.task_type in [TaskType.MULTICLASS, TaskType.BINARY] else None

        if self.loss_per_sample is not None:
            loss_per_sample = self.loss_per_sample[list(dataset.data.index)]
        else:
            loss_per_sample = calculate_per_sample_loss(context.model, context.task_type, dataset,
                                                        self.classes_index_order)
        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=True)
        if len(dataset.features) < 2:
            raise DeepchecksNotSupportedError('Check requires data to have at least two features in order to run.')
        encoded_dataset = self._target_encode_categorical_features_fill_na(dataset)
        dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=predictions, y_proba_test=y_proba,
                                  validate_data_on_predict=False)

        relevant_features = encoded_dataset.cat_features + encoded_dataset.numerical_features
        if context.feature_importance is not None:
            feature_rank = context.feature_importance.sort_values(ascending=False).keys()
            feature_rank = np.asarray([col for col in feature_rank if col in relevant_features], dtype='object')
        else:
            feature_rank = np.asarray(relevant_features, dtype='object')

        scorer = context.get_single_scorer(self.user_scorer)
        weak_segments = self._weak_segments_search(dummy_model, encoded_dataset, feature_rank,
                                                   loss_per_sample, scorer)
        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         'segments. Try increasing n_samples or supply additional features.')
        avg_score = round(scorer(dummy_model, encoded_dataset), 3)

        display = self._create_heatmap_display(dummy_model, encoded_dataset, weak_segments, avg_score,
                                               scorer) if context.with_display else []

        for idx, segment in weak_segments.copy().iterrows():
            for feature in ['Feature1', 'Feature2']:
                if segment[feature] in encoded_dataset.cat_features:
                    weak_segments[f'{feature} range'][idx] = \
                        self._format_partition_vec_for_display(segment[f'{feature} range'], segment[feature], None)[0]

        display_msg = 'Showcasing intersections of features with weakest detected segments.<br> The full list of ' \
                      'weak segments can be observed in the check result value. '
        return CheckResult({'weak_segments_list': weak_segments, 'avg_score': avg_score, 'scorer_name': scorer.name},
                           display=[display_msg, DisplayMap(display)])

    def _target_encode_categorical_features_fill_na(self, dataset: Dataset) -> Dataset:
        values_mapping = defaultdict(list)  # mapping of per feature of original values to their encoded value
        df_aggregated = default_fill_na_per_column_type(dataset.features_columns.copy(), dataset.cat_features)
        for col in dataset.cat_features:
            categories_to_mask = [k for k, v in df_aggregated[col].value_counts().items() if
                                  v / dataset.n_samples < self.categorical_aggregation_threshold]
            df_aggregated.loc[np.isin(df_aggregated[col], categories_to_mask), col] = 'Other'

        if len(dataset.cat_features) > 0:
            t_encoder = TargetEncoder(cols=dataset.cat_features)
            df_encoded = t_encoder.fit_transform(df_aggregated, dataset.label_col)
            for col in dataset.cat_features:
                values_mapping[col] = pd.concat([df_encoded[col], df_aggregated[col]], axis=1).drop_duplicates()
        else:
            df_encoded = df_aggregated
        self.encoder_mapping = values_mapping
        return Dataset(df_encoded, cat_features=dataset.cat_features, label=dataset.label_col)

    def _create_heatmap_display(self, dummy_model, encoded_dataset, weak_segments, avg_score, scorer):
        display_tabs = {}
        data = encoded_dataset.data
        idx = -1
        while len(display_tabs.keys()) < self.n_to_show and idx + 1 < len(weak_segments):
            idx += 1
            segment = weak_segments.iloc[idx, :]
            feature1 = data[segment['Feature1']]

            if segment['Feature2'] != '':
                feature2 = data[segment['Feature2']]
                segments_f1 = partition_numeric_feature_around_segment(feature1, segment['Feature1 range'])
                segments_f2 = partition_numeric_feature_around_segment(feature2, segment['Feature2 range'])
            else:
                feature2 = pd.Series(np.ones(len(feature1)))
                segments_f1 = partition_numeric_feature_around_segment(feature1, segment['Feature1 range'], 7)
                segments_f2 = [0, 2]

            scores = np.empty((len(segments_f2) - 1, len(segments_f1) - 1), dtype=float)
            counts = np.empty((len(segments_f2) - 1, len(segments_f1) - 1), dtype=int)
            for f1_idx in range(len(segments_f1) - 1):
                for f2_idx in range(len(segments_f2) - 1):
                    segment_data = data[
                        np.asarray(feature1.between(segments_f1[f1_idx], segments_f1[f1_idx + 1])) * np.asarray(
                            feature2.between(segments_f2[f2_idx], segments_f2[f2_idx + 1]))]
                    if segment_data.empty:
                        scores[f2_idx, f1_idx] = np.NaN
                        counts[f2_idx, f1_idx] = 0
                    else:
                        scores[f2_idx, f1_idx] = scorer.run_on_data_and_label(dummy_model, segment_data,
                                                                              segment_data[encoded_dataset.label_name])
                        counts[f2_idx, f1_idx] = len(segment_data)

            f1_labels = self._format_partition_vec_for_display(segments_f1, segment['Feature1'])
            f2_labels = self._format_partition_vec_for_display(segments_f2, segment['Feature2'])

            scores_text = [[0] * scores.shape[1] for _ in range(scores.shape[0])]
            counts = np.divide(counts, len(data))
            for i in range(len(f2_labels)):
                for j in range(len(f1_labels)):
                    score = scores[i, j]
                    if not np.isnan(score):
                        scores_text[i][j] = f'{format_number(score)}\n({format_percent(counts[i, j])})'
                    elif counts[i, j] == 0:
                        scores_text[i][j] = ''
                    else:
                        scores_text[i][j] = f'{score}\n({format_percent(counts[i, j])})'

            # Plotly FigureWidget have bug with numpy nan, so replacing with python None
            scores = scores.astype(np.object)
            scores[np.isnan(scores.astype(np.float_))] = None

            labels = dict(x=segment['Feature1'], y=segment['Feature2'], color=f'{scorer.name} score')
            fig = px.imshow(scores, x=f1_labels, y=f2_labels, labels=labels, color_continuous_scale='rdylgn')
            fig.update_traces(text=scores_text, texttemplate='%{text}')
            fig.update_layout(
                title=f'{scorer.name} score (percent of data) {segment["Feature1"]} vs {segment["Feature2"]}',
                height=600
            )

            msg = f'Check ran on {encoded_dataset.n_samples} data samples. Average {scorer.name} ' \
                  f'score is {format_number(avg_score)}.'
            display_tabs[f'{segment["Feature1"]} vs {segment["Feature2"]}'] = [fig, msg]

        return display_tabs

    def _weak_segments_search(self, dummy_model, encoded_dataset, feature_rank_for_search, loss_per_sample, scorer):
        """Search for weak segments based on scorer."""
        weak_segments = pd.DataFrame(
            columns=[f'{scorer.name} score', 'Feature1', 'Feature1 range', 'Feature2', 'Feature2 range', '% of data'])
        for i in range(min(len(feature_rank_for_search), self.n_top_features)):
            for j in range(i + 1, min(len(feature_rank_for_search), self.n_top_features)):
                feature1, feature2 = feature_rank_for_search[[i, j]]
                weak_segment_score, weak_segment_filter = self._find_weak_segment(dummy_model, encoded_dataset,
                                                                                  [feature1, feature2], scorer,
                                                                                  loss_per_sample)
                if weak_segment_score is None or len(weak_segment_filter.filters) == 0:
                    continue
                data_size = 100 * weak_segment_filter.filter(encoded_dataset.data).shape[0] / encoded_dataset.n_samples
                filters = weak_segment_filter.filters
                if len(filters.keys()) == 1:
                    weak_segments.loc[len(weak_segments)] = [weak_segment_score, list(filters.keys())[0],
                                                             tuple(list(filters.values())[0]), '',
                                                             None, data_size]
                else:
                    weak_segments.loc[len(weak_segments)] = [weak_segment_score, feature1,
                                                             tuple(filters[feature1]), feature2,
                                                             tuple(filters[feature2]), data_size]

        return weak_segments.drop_duplicates().sort_values(f'{scorer.name} score')

    def _find_weak_segment(self, dummy_model, dataset, features_for_segment, scorer: DeepcheckScorer, loss_per_sample):
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
                leaf_score = scorer.run_on_data_and_label(dummy_model, leaf_data, leaf_data[dataset.label_name])
                if leaf_score < min_score:
                    min_score, min_score_leaf_filter = leaf_score, leaf_filter
            return min_score, min_score_leaf_filter

        def neg_worst_segment_score(clf: DecisionTreeRegressor, x, y) -> float:  # pylint: disable=unused-argument
            return -get_worst_leaf_filter(clf.tree_)[0]

        grid_searcher = GridSearchCV(DecisionTreeRegressor(), scoring=neg_worst_segment_score,
                                     param_grid=search_space, n_jobs=-1, cv=3)
        try:
            grid_searcher.fit(dataset.features_columns[features_for_segment], loss_per_sample)
            segment_score, segment_filter = get_worst_leaf_filter(grid_searcher.best_estimator_.tree_)
        except ValueError:
            return None, None

        return segment_score, segment_filter

    def _format_partition_vec_for_display(self, partition_vec: np.array, feature_name: str,
                                          seperator: Union[str, None] = '<br>') -> List[Union[List, str]]:
        """Format partition vector for display. If seperator is None returns a list instead of a string."""
        if feature_name == '':
            return ['']

        result = []
        if feature_name in self.encoder_mapping.keys():
            feature_map_df = self.encoder_mapping[feature_name]
            encodings = feature_map_df.iloc[:, 0]
            for lower, upper in zip(partition_vec[:-1], partition_vec[1:]):
                if lower == partition_vec[0]:
                    values_in_range = np.where(np.logical_and(encodings >= lower, encodings <= upper))[0]
                else:
                    values_in_range = np.where(np.logical_and(encodings > lower, encodings <= upper))[0]
                if seperator is None:
                    result.append(feature_map_df.iloc[values_in_range, 1].to_list())
                else:
                    result.append(seperator.join([str(x) for x in feature_map_df.iloc[values_in_range, 1]]))

        else:
            for lower, upper in zip(partition_vec[:-1], partition_vec[1:]):
                result.append(f'({format_number(lower)}, {format_number(upper)}]')
            result[0] = '[' + result[0][1:]

        return result

    def add_condition_segments_relative_performance_greater_than(self, max_ratio_change: float = 0.20):
        """Add condition - check that the score of the weakest segment is greater than supplied relative threshold.

        Parameters
        ----------
        max_ratio_change : float , default: 0.20
            maximal ratio of change allowed between the average score and the score of the weakest segment.
        """

        def condition(result: Dict) -> ConditionResult:
            weakest_segment_score = result['weak_segments_list'].iloc[0, 0]
            msg = f'Found a segment with {result["scorer_name"]} score of {format_number(weakest_segment_score, 3)} ' \
                  f'in comparison to an average score of {format_number(result["avg_score"], 3)} in sampled data.'
            if result['avg_score'] > 0 and weakest_segment_score > (1 - max_ratio_change) * result['avg_score']:
                return ConditionResult(ConditionCategory.PASS, msg)
            elif result['avg_score'] < 0 and weakest_segment_score > (1 + max_ratio_change) * result['avg_score']:
                return ConditionResult(ConditionCategory.PASS, msg)
            else:
                return ConditionResult(ConditionCategory.WARN, msg)

        return self.add_condition(f'The relative performance of weakest segment is greater than '
                                  f'{format_percent(1 - max_ratio_change)} of average model performance.', condition)
