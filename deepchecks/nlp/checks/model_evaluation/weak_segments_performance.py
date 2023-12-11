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
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from deepchecks.core import CheckResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, NotEnoughSamplesError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.utils.weak_segments import get_relevant_data_table
from deepchecks.tabular.context import _DummyModel
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.single_sample_metrics import calculate_neg_cross_entropy_per_sample
from deepchecks.utils.typing import Hashable

__all__ = ['MetadataSegmentsPerformance', 'PropertySegmentsPerformance']

MIN_TEXT_SAMPLES = 10  # Min samples to calculate weak segments performance


class WeakSegmentsAbstractText(SingleDatasetCheck, WeakSegmentAbstract):
    """Check the performance of the model on different segments of the data."""

    def __init__(self, segment_by: str, columns: Union[Hashable, List[Hashable], None],
                 ignore_columns: Union[Hashable, List[Hashable], None], n_top_features: Optional[int],
                 segment_minimum_size_ratio: float, max_categories_weak_segment: Optional[int],
                 alternative_scorer: Dict[str, Union[str, Callable]],
                 score_per_sample: Union[np.ndarray, pd.Series, None], n_samples: int,
                 categorical_aggregation_threshold: float, n_to_show: int,
                 multiple_segments_per_feature: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.segment_by = segment_by
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_features = n_top_features
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.max_categories_weak_segment = max_categories_weak_segment
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.score_per_sample = score_per_sample
        self.alternative_scorer = alternative_scorer if alternative_scorer else None
        self.categorical_aggregation_threshold = categorical_aggregation_threshold
        self.multiple_segments_per_feature = multiple_segments_per_feature

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        context.raise_if_token_classification_task(self)

        text_data = context.get_data_by_kind(dataset_kind)
        text_data = text_data.sample(self.n_samples, random_state=context.random_state, drop_na_label=True)

        if text_data.n_samples < MIN_TEXT_SAMPLES:
            raise NotEnoughSamplesError(f'Not enough samples to find weak {self.segment_by} segments.'
                                        f' Minimum {MIN_TEXT_SAMPLES} samples required.')
        features, cat_features = get_relevant_data_table(text_data, data_type=self.segment_by,
                                                         columns=self.columns, ignore_columns=self.ignore_columns,
                                                         n_top_features=self.n_top_features)

        # Decide which scorer and score_per_sample to use in the algorithm run
        is_multilabel = text_data.is_multi_label_classification()
        if is_multilabel:
            if self.alternative_scorer is None:
                self.alternative_scorer = {'F1 Macro': 'f1_macro'}

            # TODO: make weak segments work with multilabel directly without reducing to single dimension
            # For multilabel, we reduce the label to a single dimension using TruncatedSVD, which is better in handling
            # dimensionality reduction of sparse matrices
            label = TruncatedSVD(1).fit_transform(text_data.label).squeeze()
            original_label = [list(x) for x in text_data.label]
        else:
            label = text_data.label
            original_label = text_data.label
        encoded_dataset = self._target_encode_categorical_features_fill_na(features, label,
                                                                           cat_features,
                                                                           is_cat_label=not is_multilabel)

        if self.score_per_sample is not None:
            score_per_sample = self.score_per_sample[list(features.index)]
            scorer, dummy_model = None, None
            avg_score = round(score_per_sample.mean(), 3)
        else:
            predictions = context.model.predict(text_data)
            if not hasattr(context.model, 'predict_proba'):
                raise DeepchecksNotSupportedError(
                    'Predicted probabilities not supplied. The weak segment checks relies'
                    ' on cross entropy error that requires predicted probabilities, '
                    'rather than only predicted classes.')
            y_proba = context.model.predict_proba(text_data)
            score_per_sample = calculate_neg_cross_entropy_per_sample(text_data.label, np.asarray(y_proba),
                                                                      is_multilabel=is_multilabel,
                                                                      model_classes=context.model_classes)
            dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=predictions, y_proba_test=y_proba,
                                      validate_data_on_predict=False)
            scorer = context.get_single_scorer(self.alternative_scorer)
            avg_score = round(scorer.run_on_data_and_label(dummy_model, encoded_dataset.features_columns,
                                                           original_label), 3)

        # Running the logic
        weak_segments = self._weak_segments_search(data=encoded_dataset.data, score_per_sample=score_per_sample,
                                                   label_col=pd.Series(original_label, index=score_per_sample.index),
                                                   feature_rank_for_search=np.asarray(encoded_dataset.features),
                                                   dummy_model=dummy_model, scorer=scorer,
                                                   multiple_segments_per_feature=self.multiple_segments_per_feature)

        if len(weak_segments) == 0:
            display_msg = 'WeakSegmentsPerformance was unable to train an error model to find weak segments.' \
                          f'Try supplying additional {self.segment_by}.'
            return CheckResult(value={'message': display_msg}, display=[display_msg])

        if context.with_display:
            display = self._create_heatmap_display(data=encoded_dataset.data, weak_segments=weak_segments,
                                                   score_per_sample=score_per_sample,
                                                   avg_score=avg_score, label_col=original_label,
                                                   dummy_model=dummy_model, scorer=scorer)
        else:
            display = []

        check_result_value = self._generate_check_result_value(weak_segments, cat_features, avg_score)
        display_msg = f'Showcasing intersections of {self.segment_by} with weakest detected segments.<br> The full ' \
                      'list of weak segments can be observed in the check result value. '
        return CheckResult(value=check_result_value,
                           display=[display_msg, DisplayMap(display)])


class PropertySegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the text properties - which are features extracted from the text, such as "language" and
    "number of words". For more on properties, see the `NLP Properties Guide
    <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_properties.html>`_.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    properties : Union[Hashable, List[Hashable]] , default: None
        Properties to check, if none are given checks all properties except ignored ones.
    ignore_properties : Union[Hashable, List[Hashable]] , default: None
        Properties to ignore, if none given checks based on properties variable
    n_top_properties : Optional[int] , default: 10
        Number of properties to use for segment search. Selected at random.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    max_categories_weak_segment: Optional[int] , default: None
        Maximum number of categories that can be included in a weak segment per categorical feature.
        If None, the number of categories is not limited.
    alternative_scorer : Dict[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    score_per_sample: Optional[np.array, pd.Series, None], default: None
        Score per sample are required to detect relevant weak segments. Should follow the convention that a sample with
        a higher score mean better model performance on that sample. If provided, the check will also use provided
        score per sample as a scoring function for segments.
        if None the check calculates score per sample by via neg cross entropy for classification.
    n_samples : int , default: 5_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    multiple_segments_per_property : bool , default: False
        If True, will allow the same property to be a segmenting feature in multiple segments,
        otherwise each property can appear in one segment at most.
    """

    def __init__(self,
                 properties: Union[Hashable, List[Hashable], None] = None,
                 ignore_properties: Union[Hashable, List[Hashable], None] = None,
                 n_top_properties: Optional[int] = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 max_categories_weak_segment: Optional[int] = None,
                 alternative_scorer: Dict[str, Union[str, Callable]] = None,
                 score_per_sample: Union[np.ndarray, pd.Series, None] = None,
                 n_samples: int = 5_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 multiple_segments_per_property: bool = False,
                 **kwargs):
        super().__init__(segment_by='properties',
                         columns=properties,
                         ignore_columns=ignore_properties,
                         n_top_features=n_top_properties,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         max_categories_weak_segment=max_categories_weak_segment,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         score_per_sample=score_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         multiple_segments_per_feature=multiple_segments_per_property,
                         **kwargs)


class MetadataSegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the metadata - which is data that is not part of the text, but is related to it,
    such as "user_id" and "user_age". For more on metadata, see the `NLP Metadata Guide
    <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_metadata.html>`_.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : Optional[int] , default: 10
        Number of columns to use for segment search. Selected at random.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    max_categories_weak_segment: Optional[int] , default: None
        Maximum number of categories that can be included in a weak segment per categorical feature.
        If None, the number of categories is not limited.
    alternative_scorer : Dict[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    score_per_sample: Union[np.array, pd.Series, None], default: None
        Score per sample are required to detect relevant weak segments. Should follow the convention that a sample with
        a higher score mean better model performance on that sample. If provided, the check will also use provided
        score per sample as a scoring function for segments.
        if None the check calculates score per sample by via neg cross entropy for classification.
    n_samples : int , default: 5_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    multiple_segments_column : bool , default: True
        If True, will allow the same metadata column to be a segmenting column in multiple segments,
        otherwise each metadata column can appear in one segment at most.
    """

    def __init__(self,
                 columns: Union[Hashable, List[Hashable], None] = None,
                 ignore_columns: Union[Hashable, List[Hashable], None] = None,
                 n_top_columns: Optional[int] = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 max_categories_weak_segment: Optional[int] = None,
                 alternative_scorer: Dict[str, Union[str, Callable]] = None,
                 score_per_sample: Union[np.ndarray, pd.Series, None] = None,
                 n_samples: int = 5_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 multiple_segments_column: bool = True,
                 **kwargs):
        super().__init__(segment_by='metadata',
                         columns=columns,
                         ignore_columns=ignore_columns,
                         n_top_features=n_top_columns,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         max_categories_weak_segment=max_categories_weak_segment,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         score_per_sample=score_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         multiple_segments_per_feature=multiple_segments_column,
                         **kwargs)
