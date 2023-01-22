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

from typing import TYPE_CHECKING, Callable, Dict, List, Union

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.docref import doclink
from deepchecks.utils.performance.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.single_sample_metrics import calculate_per_sample_loss
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
            loss_per_sample: Union[np.ndarray, pd.Series, None] = None,
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
        self.alternative_scorer = alternative_scorer if alternative_scorer else None
        self.categorical_aggregation_threshold = categorical_aggregation_threshold

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset.assert_features()
        dataset = dataset.sample(self.n_samples, random_state=self.random_state).drop_na_labels()
        predictions = context.model.predict(dataset.features_columns)
        if context.task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            if not hasattr(context.model, 'predict_proba'):
                raise DeepchecksNotSupportedError('Predicted probabilities not supplied. The weak segment checks relies'
                                                  ' on log loss error that requires predicted probabilities, rather'
                                                  ' than only predicted classes.')
            y_proba = context.model.predict_proba(dataset.features_columns)
            # If proba shape does not match label, raise error
            if y_proba.shape[1] != len(context.model_classes):
                raise DeepchecksValueError(
                    f'Predicted probabilities shape {y_proba.shape} does not match the number of classes found in'
                    f' the labels {context.model_classes}.')
        else:
            y_proba = None

        if self.loss_per_sample is not None:
            loss_per_sample = self.loss_per_sample[list(dataset.data.index)]
        else:
            loss_per_sample = calculate_per_sample_loss(context.model, context.task_type, dataset,
                                                        context.model_classes)
        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=True)
        if len(dataset.features) < 2:
            raise DeepchecksNotSupportedError('Check requires data to have at least two features in order to run.')
        encoded_dataset = self._target_encode_categorical_features_fill_na(dataset, context.observed_classes)
        dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=predictions, y_proba_test=y_proba,
                                  validate_data_on_predict=False)

        relevant_features = encoded_dataset.cat_features + encoded_dataset.numerical_features
        if context.feature_importance is not None:
            feature_rank = context.feature_importance.sort_values(ascending=False).keys()
            feature_rank = np.asarray([col for col in feature_rank if col in relevant_features], dtype='object')
        else:
            feature_rank = np.asarray(relevant_features, dtype='object')

        scorer = context.get_single_scorer(self.alternative_scorer)
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
