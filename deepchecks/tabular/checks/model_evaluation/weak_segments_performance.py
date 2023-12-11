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
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.docref import doclink
from deepchecks.utils.single_sample_metrics import calculate_neg_cross_entropy_per_sample, calculate_neg_mse_per_sample
from deepchecks.utils.typing import Hashable

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['WeakSegmentsPerformance']


class WeakSegmentsPerformance(SingleDatasetCheck, WeakSegmentAbstract):
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
    n_top_features : Optional[int] , default: 10
        Number of features to use for segment search. Top columns are selected based on feature importance.
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
        if None the check calculates score per sample by via neg cross entropy for classification and
        neg MSE for regression.
    loss_per_sample: Union[np.array, pd.Series, None], default: None
        Deprecated, please use score_per_sample instead.
    n_samples : int , default: 10_000
        number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    random_state : int, default: 42
        random seed for all check internals.
    multiple_segments_per_feature : bool , default: True
        If True, will allow the same feature to be a segmenting feature in multiple segments,
        otherwise each feature can appear in one segment at most.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_top_features: Optional[int] = 10,
            segment_minimum_size_ratio: float = 0.05,
            max_categories_weak_segment: Optional[int] = None,
            alternative_scorer: Dict[str, Union[str, Callable]] = None,
            loss_per_sample: Union[np.ndarray, pd.Series, None] = None,
            score_per_sample: Union[np.ndarray, pd.Series, None] = None,
            n_samples: int = 10_000,
            categorical_aggregation_threshold: float = 0.05,
            n_to_show: int = 3,
            random_state: int = 42,
            multiple_segments_per_feature: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if loss_per_sample is not None and score_per_sample is None:
            warnings.warn(f'{self.__class__.__name__}: loss_per_sample is deprecated. '
                          f'Please use score_per_sample instead.', DeprecationWarning)
            score_per_sample = - np.asarray(loss_per_sample)
        if score_per_sample is not None and alternative_scorer:
            raise DeepchecksValueError('Cannot use both score_per_sample and alternative_scorer')
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_features = n_top_features
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.max_categories_weak_segment = max_categories_weak_segment
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state
        self.score_per_sample = score_per_sample
        self.loss_per_sample = loss_per_sample
        self.alternative_scorer = alternative_scorer
        self.categorical_aggregation_threshold = categorical_aggregation_threshold
        self.multiple_segments_per_feature = multiple_segments_per_feature

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.sample(self.n_samples, random_state=self.random_state).drop_na_labels()
        dataset_subset = dataset.select(self.columns, self.ignore_columns, keep_label=True)
        if len(dataset.features) < 2:
            raise DeepchecksNotSupportedError('Check requires data to have at least two features in order to run.')

        # Decide which scorer and score_per_sample to use in the algorithm run
        features_data = dataset_subset.features_columns[dataset_subset.numerical_features + dataset_subset.cat_features]
        encoded_dataset = self._target_encode_categorical_features_fill_na(features_data,
                                                                           dataset.label_col,
                                                                           dataset_subset.cat_features,
                                                                           context.task_type != TaskType.REGRESSION)

        if self.score_per_sample is not None:
            score_per_sample = self.score_per_sample[list(dataset.data.index)]
            scorer, dummy_model = None, None
            avg_score = round(score_per_sample.mean(), 3)
        else:
            predictions = context.model.predict(dataset.features_columns)
            if context.task_type == TaskType.REGRESSION:
                y_proba = None
                score_per_sample = calculate_neg_mse_per_sample(dataset_subset.label_col, predictions)
            elif context.task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
                if not hasattr(context.model, 'predict_proba'):
                    raise DeepchecksNotSupportedError(
                        'Predicted probabilities not supplied. The weak segment checks relies'
                        ' on cross entropy error that requires predicted probabilities, '
                        'rather than only predicted classes.')
                y_proba = context.model.predict_proba(dataset.features_columns)
                score_per_sample = calculate_neg_cross_entropy_per_sample(dataset.label_col, y_proba,
                                                                          context.model_classes)

            dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=predictions, y_proba_test=y_proba,
                                      validate_data_on_predict=False)
            scorer = context.get_single_scorer(self.alternative_scorer)
            avg_score = round(scorer(dummy_model, encoded_dataset), 3)

        # Calculating feature rank
        relevant_features = dataset_subset.cat_features + dataset_subset.numerical_features
        if context.feature_importance is not None:
            feature_rank = context.feature_importance.sort_values(ascending=False).keys()
            feature_rank = np.asarray([col for col in feature_rank if col in relevant_features], dtype='object')
        else:
            feature_rank = np.asarray(relevant_features, dtype='object')

        # Running the logic
        weak_segments = self._weak_segments_search(data=encoded_dataset.data, score_per_sample=score_per_sample,
                                                   label_col=dataset_subset.label_col,
                                                   feature_rank_for_search=feature_rank,
                                                   dummy_model=dummy_model, scorer=scorer,
                                                   multiple_segments_per_feature=self.multiple_segments_per_feature)

        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         'segments. Try increasing n_samples or supply additional features.')

        if context.with_display:
            display = self._create_heatmap_display(data=encoded_dataset.data, weak_segments=weak_segments,
                                                   score_per_sample=score_per_sample,
                                                   avg_score=avg_score, label_col=dataset_subset.label_col,
                                                   dummy_model=dummy_model, scorer=scorer)
        else:
            display = []

        check_result_value = self._generate_check_result_value(weak_segments, dataset_subset.cat_features, avg_score)
        display_msg = 'Showcasing intersections of features with weakest detected segments.<br> The full list of ' \
                      'weak segments can be observed in the check result value. '
        return CheckResult(value=check_result_value,
                           display=[display_msg, DisplayMap(display)])

    def config(self, include_version: bool = True, include_defaults: bool = True) -> 'CheckConfig':
        """Return checks instance config."""
        if isinstance(self.alternative_scorer, dict):
            for k, v in self.alternative_scorer.items():
                if callable(v):
                    reference = doclink(
                        'supported-metrics-by-string',
                        template='For a list of built-in scorers please refer to {link}. ',
                    )
                    raise ValueError(
                        'Only built-in scorers are allowed when serializing check instances. '
                        f'{reference}Scorer name: {k}'
                    )
        return super().config(include_version, include_defaults=include_defaults)
