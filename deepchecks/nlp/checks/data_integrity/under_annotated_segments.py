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
"""Module of weak segments performance check."""
from typing import Dict, List, Union

import pandas as pd

from deepchecks import ConditionCategory, ConditionResult
from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.utils.weak_segments import get_relevant_data_table
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['UnderAnnotatedMetaDataSegments', 'UnderAnnotatedPropertySegments']


class UnderAnnotatedSegments(SingleDatasetCheck, WeakSegmentAbstract):
    """Check for under annotated data segments."""

    def __init__(self, segment_by: str, columns: Union[Hashable, List[Hashable], None],
                 ignore_columns: Union[Hashable, List[Hashable], None], n_top_features: int,
                 segment_minimum_size_ratio: float, n_samples: int,
                 categorical_aggregation_threshold: float, n_to_show: int, **kwargs):
        super().__init__(**kwargs)
        self.segment_by = segment_by
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_features = n_top_features
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.categorical_aggregation_threshold = categorical_aggregation_threshold

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        context.raise_if_token_classification_task(self)
        context.raise_if_multi_label_task(self)

        text_data = context.get_data_by_kind(dataset_kind)
        text_data = text_data.sample(self.n_samples, random_state=context.random_state)

        features, cat_features = get_relevant_data_table(text_data, data_type=self.segment_by,
                                                         columns=self.columns, ignore_columns=self.ignore_columns,
                                                         n_top_features=self.n_top_features)

        encoded_dataset = self._target_encode_categorical_features_fill_na(features, text_data.label,
                                                                           cat_features)

        score_per_sample = pd.Series([1 - pd.isna(x) for x in text_data.label], index=encoded_dataset.data.index)
        avg_score = round(score_per_sample.mean(), 3)
        weak_segments = self._weak_segments_search(data=encoded_dataset.features_columns,
                                                   score_per_sample=score_per_sample,
                                                   scorer_name='Annotation Ratio')

        if len(weak_segments) == 0:
            raise DeepchecksProcessError('Check was unable to find under annotated segments. This is expected if '
                                         'your data is well annotated. If this is not the case, try increasing '
                                         f'n_samples or supply more {self.segment_by}.')

        check_result_value = self._generate_check_result_value(weak_segments, cat_features, avg_score)
        display_msg = f'Showcasing intersections of {self.segment_by} with most under annotated segments.<br> The ' \
                      'full list of under annotated segments can be observed in the check result value. '
        return CheckResult(value=check_result_value, display=[display_msg, 'place holder for figure'])

    def add_condition_segments_annotation_ratio_greater_than(self, threshold: float = 0.70):
        """Add condition - check that the in all segments annotation ratio is above the provided threshold.

        Parameters
        ----------
        threshold : float , default: 0.20
            maximal ratio of change allowed between the average score and the score of the weakest segment.
        """

        def condition(result: Dict) -> ConditionResult:
            weakest_segment_score = result['weak_segments_list'].iloc[0, 0]
            msg = f'Most under annotated segment has annotation ratio of {format_percent(weakest_segment_score)}.'
            if weakest_segment_score > threshold:
                return ConditionResult(ConditionCategory.PASS, msg)
            else:
                return ConditionResult(ConditionCategory.FAIL, msg)

        return self.add_condition(f'In all segments annotation ratio should be greater than '
                                  f'{format_percent(threshold)}.', condition)


class UnderAnnotatedPropertySegments(UnderAnnotatedSegments):
    """Search for under annotated data segments.

    The check is designed to help you easily identify under annotated segments of your data. The segments are
    based on the text properties - which are features extracted from the text, such as "language" and
    "number of words".

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    properties : Union[Hashable, List[Hashable]] , default: None
        Properties to check, if none are given checks all properties except ignored ones.
    ignore_properties : Union[Hashable, List[Hashable]] , default: None
        Properties to ignore, if none given checks based on properties variable
    n_top_properties : int , default: 10
        Number of properties to use for segment search. Top properties are selected based on feature importance.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    n_samples : int , default: 10_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    """

    def __init__(self,
                 properties: Union[Hashable, List[Hashable], None] = None,
                 ignore_properties: Union[Hashable, List[Hashable], None] = None,
                 n_top_properties: int = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 n_samples: int = 10_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 **kwargs):
        super().__init__(segment_by='properties',
                         columns=properties,
                         ignore_columns=ignore_properties,
                         n_top_features=n_top_properties,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         **kwargs)


class UnderAnnotatedMetaDataSegments(UnderAnnotatedSegments):
    """Search for under annotated data segments.

    The check is designed to help you easily identify under annotated segments of your data. The segments are
    based on the metadata - which is data that is not part of the text, but is related to it,
    such as "user_id" and "user_age".

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : int , default: 10
        Number of features to use for segment search. Top columns are selected based on feature importance.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    n_samples : int , default: 10_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    """

    def __init__(self,
                 columns: Union[Hashable, List[Hashable], None] = None,
                 ignore_columns: Union[Hashable, List[Hashable], None] = None,
                 n_top_columns: int = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 n_samples: int = 10_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 **kwargs):
        super().__init__(segment_by='metadata',
                         columns=columns,
                         ignore_columns=ignore_columns,
                         n_top_features=n_top_columns,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         **kwargs)