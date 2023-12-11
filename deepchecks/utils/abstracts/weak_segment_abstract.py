# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains common methods for weak segment performance checks."""
import abc
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from category_encoders import TargetEncoder
from packaging import version
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from deepchecks import ConditionCategory, ConditionResult
from deepchecks.tabular import Dataset
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.metric_utils.scorers import DeepcheckScorer
from deepchecks.utils.dataframes import default_fill_na_per_column_type
from deepchecks.utils.performance.partition import (DeepchecksFilter, convert_tree_leaves_into_filters,
                                                    partition_numeric_feature_around_segment)
from deepchecks.utils.strings import format_number, format_percent


class WeakSegmentAbstract(abc.ABC):
    """Abstract class with common methods to be inherited by WeakSegmentsPerformance checks."""

    n_top_features: int = 5
    n_to_show: int = 3
    categorical_aggregation_threshold: float = 0.05
    min_category_size_ratio: float = 0.01
    segment_minimum_size_ratio: float = 0.05
    max_categories_weak_segment: Optional[int] = None
    random_state: int = 42
    add_condition: Callable[..., Any]

    def _target_encode_categorical_features_fill_na(self, data: pd.DataFrame, label_col: pd.Series,
                                                    cat_features: List[str], is_cat_label: bool = True) -> Dataset:
        values_mapping = defaultdict(list)  # mapping of per feature of original values to their encoded value
        label_col = pd.Series(label_col, index=data.index)
        df_aggregated = default_fill_na_per_column_type(data, cat_features)
        cat_features = [col for col in cat_features if col in df_aggregated.columns]
        # Merging small categories into Other
        for col in cat_features:
            categories_to_mask = [k for k, v in df_aggregated[col].value_counts().items() if
                                  v / data.shape[0] < self.categorical_aggregation_threshold]
            df_aggregated.loc[np.isin(df_aggregated[col], categories_to_mask), col] = 'Other'

        # Target encoding of categorical features based on label col (when unavailable we use loss_per_sample)
        if len(cat_features) > 0:
            t_encoder = TargetEncoder(cols=cat_features)
            if is_cat_label:
                label_no_none = label_col.astype('object').fillna('None')
                label_as_int = pd.Categorical(label_no_none, categories=sorted(label_no_none.unique())).codes
            else:
                label_as_int = pd.cut(label_col.astype('float64').fillna(label_col.mean()), bins=10, labels=False)
            df_encoded = t_encoder.fit_transform(df_aggregated, pd.Series(label_as_int, index=df_aggregated.index))
            # Convert categorical features to ordinal based on their encoded values and store the mapping
            for col in cat_features:
                df_encoded[col] = df_encoded[col].apply(sorted(df_encoded[col].unique()).index)
                mapping = pd.concat([df_encoded[col], df_aggregated[col]], axis=1).drop_duplicates()
                mapping.columns = ['encoded_value', 'original_category']
                values_mapping[col] = mapping.sort_values(by='encoded_value')
        else:
            df_encoded = df_aggregated
        self.encoder_mapping = values_mapping
        return Dataset(df_encoded, cat_features=cat_features, label=label_col)

    def _create_heatmap_display(self, data: pd.DataFrame,
                                weak_segments: pd.DataFrame, avg_score: float,
                                score_per_sample: Optional[pd.Series] = None,
                                label_col: Optional[pd.Series] = None,
                                dummy_model: Optional[_DummyModel] = None,
                                scorer: Optional[DeepcheckScorer] = None, scorer_name: Optional[str] = None):
        display_tabs = {}
        if scorer_name is None and scorer is None:
            score_title = 'Average Score Per Sample'
        else:
            score_title = scorer_name if scorer_name is not None else scorer.name + ' Score'
        temp_col = 'temp_label_column'
        if label_col is not None:  # add label col to data for filtering purposes
            data[temp_col] = label_col

        idx = -1
        while len(display_tabs.keys()) < self.n_to_show and idx + 1 < len(weak_segments):
            idx += 1
            segment = weak_segments.iloc[idx, :]
            feature1 = data[segment['Feature1']]

            # Bin the rest of the data into data segments based on the features used to create the weak segment
            if segment['Feature2'] != '':
                feature2 = data[segment['Feature2']]
                segments_f1 = partition_numeric_feature_around_segment(feature1, segment['Feature1 Range'])
                segments_f2 = partition_numeric_feature_around_segment(feature2, segment['Feature2 Range'])
            else:  # if only one feature is used to create the weak segment
                feature2 = pd.Series(np.ones(len(feature1)))
                segments_f1 = partition_numeric_feature_around_segment(feature1, segment['Feature1 Range'], 7)
                segments_f2 = [0, 2]

            # Calculate the score for each of the bins defined above
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
                        if scorer is not None and dummy_model is not None and label_col is not None:
                            scores[f2_idx, f1_idx] = scorer.run_on_data_and_label(dummy_model,
                                                                                  segment_data.drop(columns=temp_col),
                                                                                  segment_data[temp_col])
                        else:
                            scores[f2_idx, f1_idx] = score_per_sample[list(segment_data.index)].mean()
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
            scores = scores.astype(object)
            scores[np.isnan(scores.astype(float))] = None

            labels = dict(x=segment['Feature1'], y=segment['Feature2'], color=score_title)
            fig = px.imshow(scores, x=f1_labels, y=f2_labels, labels=labels, color_continuous_scale='rdylgn')
            fig.update_traces(text=scores_text, texttemplate='%{text}')
            if segment['Feature2']:
                title = f'{score_title} (percent of data)'
                tab_name = f'{segment["Feature1"]} vs {segment["Feature2"]}'
            else:
                title = f'{score_title} (percent of data)'
                tab_name = segment['Feature1']
            fig.update_layout(
                title=title,
                height=600,
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )

            msg = f'Check ran on {data.shape[0]} data samples. {score_title} on the full data set ' \
                  f'is {format_number(avg_score)}.'
            display_tabs[tab_name] = [fig, msg]

        return display_tabs

    def _weak_segments_search(self, data: pd.DataFrame, score_per_sample: pd.Series,
                              label_col: Optional[pd.Series] = None,
                              feature_rank_for_search: Optional[np.ndarray] = None,
                              dummy_model: Optional[_DummyModel] = None, scorer: Optional[DeepcheckScorer] = None,
                              scorer_name: Optional[str] = None, multiple_segments_per_feature: bool = False) \
            -> pd.DataFrame:
        """Search for weak segments based on scorer."""
        # Remove samples with NaN score per sample
        score_per_sample = score_per_sample.dropna()
        data = data.loc[score_per_sample.index]
        if label_col is not None:
            label_col = label_col.loc[score_per_sample.index]

        if scorer_name is None and scorer is None:
            score_title = 'Average Score Per Sample'
        else:
            score_title = scorer_name if scorer_name is not None else scorer.name + ' Score'
        if feature_rank_for_search is None:
            feature_rank_for_search = np.asarray(data.columns)

        weak_segments = pd.DataFrame(
            columns=[score_title, 'Feature1', 'Feature1 Range', 'Feature2', 'Feature2 Range',
                     '% of Data', 'Samples in Segment'])
        n_features = min(len(feature_rank_for_search), self.n_top_features) if self.n_top_features is not None \
            else len(feature_rank_for_search)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature1, feature2 = feature_rank_for_search[[i, j]]
                # Categorical feature come first
                if feature1 not in self.encoder_mapping and feature2 in self.encoder_mapping:
                    feature2, feature1 = feature_rank_for_search[[i, j]]
                weak_segment_score, weak_segment_filter = self._find_weak_segment(data, [feature1, feature2],
                                                                                  score_per_sample, label_col,
                                                                                  dummy_model, scorer)
                if weak_segment_score is None or len(weak_segment_filter.filters) == 0:
                    continue
                data_of_segment = weak_segment_filter.filter(data)
                data_size = round(100 * data_of_segment.shape[0] / data.shape[0], 2)
                filters = weak_segment_filter.filters
                if len(filters.keys()) == 1:
                    weak_segments.loc[len(weak_segments)] = [weak_segment_score, list(filters.keys())[0],
                                                             tuple(list(filters.values())[0]), '',
                                                             None, data_size, list(data_of_segment.index)]
                else:
                    weak_segments.loc[len(weak_segments)] = [weak_segment_score, feature1,
                                                             tuple(filters[feature1]), feature2,
                                                             tuple(filters[feature2]), data_size,
                                                             list(data_of_segment.index)]

        # Filter and adapt the weak segments results
        result = pd.DataFrame(columns=weak_segments.columns)
        used_features = set()
        for _, row in weak_segments.sort_values(score_title).iterrows():
            new_row = row.copy()
            if not multiple_segments_per_feature and \
                    (row['Feature1'] in used_features or row['Feature2'] in used_features):
                continue

            # Make sure segments based on categorical features are based only on a single category
            if self.max_categories_weak_segment is not None and row['Feature1'] in self.encoder_mapping:
                unique_values_in_range = [x for x in self.encoder_mapping[row['Feature1']]['encoded_value'].values if
                                          row['Feature1 Range'][1] > x > row['Feature1 Range'][0]]
                if len(unique_values_in_range) > self.max_categories_weak_segment:
                    subset = data.loc[new_row['Samples in Segment']]
                    value_segment_size = [len(subset[subset[row['Feature1']] == x]) for x in unique_values_in_range]
                    # If all sub segments are too small, remove feature 2 filter
                    if max(value_segment_size) < len(data) * self.segment_minimum_size_ratio and row['Feature2'] != '':
                        subset = data
                        value_segment_size = [len(data[data[row['Feature1']] == x]) for x in unique_values_in_range]
                        new_row['Feature2'] = ''
                        new_row['Feature2 Range'] = None
                    if max(value_segment_size) < len(data) * self.segment_minimum_size_ratio:
                        continue

                    value_to_use = unique_values_in_range[np.argmax(value_segment_size)]
                    subset = subset[subset[row['Feature1']] == value_to_use]
                    new_row['Samples in Segment'] = list(subset.index)
                    new_row['% of Data'] = round(100 * len(new_row['Samples in Segment']) / len(data), 2)
                    new_row['Feature1 Range'] = [value_to_use - 0.5, value_to_use + 0.5]
                    if dummy_model is not None and label_col is not None and scorer is not None:
                        new_row[score_title] = scorer.run_on_data_and_label(dummy_model,
                                                                            subset, label_col[list(subset.index)])
                    else:
                        new_row[score_title] = score_per_sample[list(subset.index)].mean()

            if self.max_categories_weak_segment is not None and \
                    new_row['Feature2'] != '' and row['Feature2'] in self.encoder_mapping:
                unique_values_in_range = [x for x in self.encoder_mapping[row['Feature2']]['encoded_value'].values if
                                          row['Feature2 Range'][1] > x > row['Feature2 Range'][0]]
                if len(unique_values_in_range) > self.max_categories_weak_segment:
                    subset = data.loc[new_row['Samples in Segment']]
                    value_segment_size = [len(subset[subset[row['Feature2']] == x]) for x in unique_values_in_range]
                    # Feature 1 cannot be empty so if feature 2 do not have a large enough segment, ignore the row
                    if max(value_segment_size) < len(data) * self.segment_minimum_size_ratio:
                        continue
                    value_to_use = unique_values_in_range[np.argmax(value_segment_size)]
                    subset = subset[subset[row['Feature2']] == value_to_use]
                    new_row['Samples in Segment'] = list(subset.index)
                    new_row['% of Data'] = round(100 * len(new_row['Samples in Segment']) / len(data), 2)
                    new_row['Feature2 Range'] = [value_to_use - 0.5, value_to_use + 0.5]
                    if dummy_model is not None and label_col is not None and scorer is not None:
                        new_row[score_title] = scorer.run_on_data_and_label(dummy_model,
                                                                            subset, label_col[list(subset.index)])
                    else:
                        new_row[score_title] = score_per_sample[list(subset.index)].mean()

            result.loc[len(result)] = new_row
            used_features.add(new_row['Feature1'])
            if new_row['Feature2'] != '':
                used_features.add(new_row['Feature2'])

        return result.sort_values(score_title).drop_duplicates(subset=['Feature1', 'Feature2'])

    def _find_weak_segment(self, data: pd.DataFrame, features_for_segment: List[str], score_per_sample: pd.Series,
                           label_col: Optional[pd.Series] = None, dummy_model: Optional[_DummyModel] = None,
                           scorer: Optional[DeepcheckScorer] = None) -> \
            Tuple[Optional[float], Optional[DeepchecksFilter]]:
        """Find weak segment based on scorer for specified features.

        In each iteration build a decision tree with a set of parameters with the goal of grouping samples with
        similar loss_per_sample values together. Then, the generated tree is values based only on the quality of
        the worst leaf in the tree (the rest are ignored). The leaf score is calculated by the scorer
        if provided, otherwise by the average score_per_sample value of the leaf.

        After all the iterations are done, the tree with the best score (the one with the worst leaf) is selected, and
        the worst leaf of it is extracted and returned as a deepchecks filter.
        """
        # Remove rows with na values in the relevant columns
        data_for_search = data[features_for_segment].dropna()
        if len(data_for_search) == 0:
            return None, None
        segment_minimum_size_ratio = self.segment_minimum_size_ratio * len(data) / len(data_for_search)
        score_per_sample_for_search = score_per_sample.loc[data_for_search.index]
        if label_col is not None:
            label_col_for_search = label_col.loc[data_for_search.index]

        if version.parse(sklearn.__version__) < version.parse('1.0.0'):
            criterion = ['mse', 'mae']
        else:
            criterion = ['squared_error', 'absolute_error']
        search_space = {
            'max_depth': [5],
            'min_weight_fraction_leaf': [segment_minimum_size_ratio],
            'min_samples_leaf': [5],
            'criterion': criterion,
            'min_impurity_decrease': [0.003],
        }

        # In a given tree finds the leaf with the worst score (the rest are ignored)
        def get_worst_leaf_filter(tree):
            leaves_filters = convert_tree_leaves_into_filters(tree, features_for_segment)
            min_score, min_score_leaf_filter = np.inf, None
            for leaf_filter in leaves_filters:
                if scorer is not None and dummy_model is not None and label_col is not None:
                    leaf_data, leaf_labels = leaf_filter.filter(data_for_search, label_col_for_search)
                    leaf_score = scorer.run_on_data_and_label(dummy_model, leaf_data, leaf_labels)
                else:  # if no scorer is provided, use the average loss_per_sample of samples in the leaf as the score
                    leaf_score = score_per_sample_for_search[list(leaf_filter.filter(data_for_search).index)].mean()

                if leaf_score < min_score:
                    min_score, min_score_leaf_filter = leaf_score, leaf_filter
            return min_score, min_score_leaf_filter

        def neg_worst_segment_score(clf: DecisionTreeRegressor, x, y) -> float:  # pylint: disable=unused-argument
            return -get_worst_leaf_filter(clf.tree_)[0]

        if hasattr(self, 'random_state'):
            random_state = self.random_state
        elif hasattr(self, 'context'):
            random_state = self.context.random_state
        else:
            random_state = None
        grid_searcher = GridSearchCV(DecisionTreeRegressor(random_state=random_state),
                                     scoring=neg_worst_segment_score, param_grid=search_space, n_jobs=-1, cv=3)
        try:
            grid_searcher.fit(data_for_search, score_per_sample_for_search)
            # Get the worst leaf filter out of the selected tree
            segment_score, segment_filter = get_worst_leaf_filter(grid_searcher.best_estimator_.tree_)
        except ValueError:
            return None, None

        return segment_score, segment_filter

    def _format_partition_vec_for_display(self, partition_vec: np.array, feature_name: str,
                                          seperator: Union[str, None] = '<br>') -> List[Union[List, str]]:
        """Format partition vector for display. If seperator is None returns a list instead of a string."""
        if feature_name == '':
            return ['']
        if not isinstance(partition_vec, np.ndarray):
            partition_vec = np.asarray(partition_vec)

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

    def _generate_check_result_value(self, weak_segments_df, cat_features: List[str], avg_score: float):
        """Generate a uniform format check result value for the different WeakSegmentsPerformance checks."""
        pd.set_option('mode.chained_assignment', None)
        weak_segments_output = weak_segments_df.copy()
        for idx, segment in weak_segments_df.iterrows():
            for feature in ['Feature1', 'Feature2']:
                if segment[feature] in cat_features:
                    weak_segments_output[f'{feature} Range'][idx] = \
                        self._format_partition_vec_for_display(segment[f'{feature} Range'], segment[feature], None)[0]

        return {'weak_segments_list': weak_segments_output, 'avg_score': avg_score}

    def add_condition_segments_relative_performance_greater_than(self, max_ratio_change: float = 0.20):
        """Add condition - check that the score of the weakest segment is greater than supplied relative threshold.

        Parameters
        ----------
        max_ratio_change : float , default: 0.20
            maximal ratio of change allowed between the average score and the score of the weakest segment.
        """

        def condition(result: Dict) -> ConditionResult:
            if 'message' in result:
                return ConditionResult(ConditionCategory.PASS, result['message'])

            weakest_segment_score = result['weak_segments_list'].iloc[0, 0]
            scorer_name = result['weak_segments_list'].columns[0].lower()
            msg = f'Found a segment with {scorer_name} of {format_number(weakest_segment_score, 3)} ' \
                  f'in comparison to an average score of {format_number(result["avg_score"], 3)} in sampled data.'
            if result['avg_score'] > 0 and weakest_segment_score > (1 - max_ratio_change) * result['avg_score']:
                return ConditionResult(ConditionCategory.PASS, msg)
            elif result['avg_score'] < 0 and weakest_segment_score > (1 + max_ratio_change) * result['avg_score']:
                return ConditionResult(ConditionCategory.PASS, msg)
            else:
                return ConditionResult(ConditionCategory.WARN, msg)

        return self.add_condition(f'The relative performance of weakest segment is greater than '
                                  f'{format_percent(1 - max_ratio_change)} of average model performance.', condition)
