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
from copy import copy
from typing import TYPE_CHECKING, Callable, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
#from imblearn.over_sampling import SMOTENC

from deepchecks.core import CheckResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.core.fix_classes import SingleDatasetCheckFixMixin
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.docref import doclink
from deepchecks.utils.performance.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.single_sample_metrics import calculate_per_sample_loss
from deepchecks.utils.typing import Hashable

# if TYPE_CHECKING:
from deepchecks.core.checks import CheckConfig, DatasetKind

__all__ = ['WeakSegmentsPerformance']


class WeakSegmentsPerformance(SingleDatasetCheck, WeakSegmentAbstract, SingleDatasetCheckFixMixin):
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

    def fix_logic(self, context: Context, check_result, dataset_kind, oversample_by: str = 'duplicates',
                  num_segment_to_fix: int = 1, oversample_factor: Union[float, str] = 'auto') -> Context:
        """Run fix.

        Parameters
        ----------
        context: Context
            The context object to fix
        check_result: CheckResult
            The check result to fix
        dataset_kind: DatasetKind
            The dataset kind to fix
        oversample_by: str
            The method to use for oversampling. Can be 'duplicates' or 'smote'
        num_segment_to_fix: int
            The number of segments to fix
        oversample_factor: Union[float, str]
            The factor to use for oversampling. If 'auto' is given, the factor will be calculated automatically.
        """
        if oversample_factor == 'auto':
            oversample_factor = 2  # TODO Temporarily set to 2, need to calculate the factor
        elif oversample_factor < 1:
            raise DeepchecksValueError('Oversample factor must be greater than 1')

        # Fix will always be done on train dataset:
        dataset = context.get_data_by_kind(DatasetKind.TRAIN)
        data = dataset.data.copy()

        weak_segments = check_result.value['weak_segments_list'].iloc[:num_segment_to_fix]

        new_i = 0

        for _, segment in weak_segments.iterrows():
            feature_1, feature_2 = segment['Feature1'], segment['Feature2']
            feature_1_range, feature_2_range = segment['Feature1 range'], segment['Feature2 range']

            feature_1_samples = get_samples_by_range(s=data[feature_1], values_range=feature_1_range,
                                                     is_categorical=feature_1 in dataset.cat_features)
            if feature_2 is not None:
                feature_2_samples = get_samples_by_range(s=data[feature_2], values_range=feature_2_range,
                                                         is_categorical=feature_2 in dataset.cat_features)
                relevant_samples = data.loc[feature_1_samples & feature_2_samples]
            else:
                relevant_samples = data.loc[feature_1_samples]

            if relevant_samples.empty:
                warnings.warn(f'No samples were found for segment, cannot fix.')
            else:
                if oversample_by == 'duplicates':
                    additional_samples = relevant_samples.sample(frac=oversample_factor - 1, replace=True,
                                                                 random_state=self.random_state)
                    additional_samples.index = [f'duplicate_{i}' for i in range(new_i, new_i + len(additional_samples))]
                    new_i += len(additional_samples)
                elif oversample_by.lower() == 'smote':
                    #TODO: Add SMOTE support python3.6
                    pass
                    # SMOTENC requires a label, we'll give the label as whether the sample is from the weak segment
                    # or not:
                    irrelevant_samples = data[~data.index.isin(relevant_samples.index)]
                    data_to_smote = pd.concat([relevant_samples, irrelevant_samples])
                    label_to_smote = pd.Series([0] * relevant_samples.shape[0] + [1] * irrelevant_samples.shape[0],
                                               index=data.index)

                    n_samples_for_smote = int(len(relevant_samples) * oversample_factor)
                    n_samples_to_add = n_samples_for_smote - len(relevant_samples)

                    # SMOTE requires categorical features to be encoded as their location in the data:
                    cat_features_for_smote = copy(dataset.cat_features)
                    if context.task_type != TaskType.REGRESSION:
                        cat_features_for_smote.append(dataset.label_name)
                    cat_indexes_for_smote = [data_to_smote.columns.get_loc(cat_feature) for cat_feature in
                                             cat_features_for_smote]

                    # smote = SMOTENC(categorical_features=cat_indexes_for_smote, random_state=self.random_state,
                    #                 sampling_strategy={0: n_samples_for_smote})
                    # additional_samples = smote.fit_resample(data_to_smote, label_to_smote)[0][-n_samples_to_add:]
                    # additional_samples.index = [f'generated_{i}' for i in
                    #                             range(new_i, new_i + n_samples_to_add)]
                    new_i += n_samples_to_add

            data = pd.concat([data, additional_samples])

        context.set_dataset_by_kind(DatasetKind.TRAIN, dataset.copy(data))
        return context

    @property
    def fix_params(self):
        """Return fix params for display."""
        return {'oversample_by': {'display': 'Oversample By',
                                  'params': ['duplicates', 'smote'],
                                  'params_display': ['Duplicating Samples', 'Using SMOTE'],
                                  'params_description': ['Duplicate samples', 'Use SMOTE to generate new samples']},
                'num_segment_to_fix': {'display': 'Number of Segments to Fix',
                                       'params': float,
                                       'params_display': 1,
                                       'min_value': 1,
                                       'params_description': 'The number of segments to fix'},
                'oversample_factor': {'display': 'Oversample Multiplier',
                                      'params': ['auto', 2, 3, 4, 5],
                                      'params_display': ['Auto', 'X2', 'X3', 'X4', 'X5'],
                                      'params_description': ['Automatically calculate the number of segments to fix',
                                                             'Oversample by 2', 'Oversample by 3', 'Oversample by 4',
                                                             'Oversample by 5']}}

    @property
    def problem_description(self):
        """Return problem description."""
        return """Duplicate data samples are present in the dataset. This can lead to overfitting and
                  decrease the performance of the model."""

    @property
    def manual_solution_description(self):
        """Return manual solution description."""
        return """Remove duplicate samples."""

    @property
    def automatic_solution_description(self):
        """Return automatic solution description."""
        return """Remove duplicate samples."""


def get_samples_by_range(s: pd.Series, values_range: Union[List, Tuple], is_categorical: bool):
    if is_categorical:
        return s[s.isin(values_range)].index
    else:
        return s[s.between(*values_range)].index
