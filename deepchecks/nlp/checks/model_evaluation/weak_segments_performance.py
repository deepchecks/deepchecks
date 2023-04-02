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
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from deepchecks.core import CheckResult
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.tabular import Dataset
from deepchecks.tabular.context import _DummyModel
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.typing import Hashable

__all__ = ['MetadataSegmentsPerformance', 'PropertySegmentsPerformance']


class WeakSegmentsAbstractText(SingleDatasetCheck, WeakSegmentAbstract):
    """Check the performance of the model on different segments of the data."""

    def __init__(self, segment_by: str, columns: Union[Hashable, List[Hashable], None],
                 ignore_columns: Union[Hashable, List[Hashable], None], n_top_features: int,
                 segment_minimum_size_ratio: float, alternative_scorer: Dict[str, Callable],
                 loss_per_sample: Union[np.ndarray, pd.Series, None], n_samples: int,
                 categorical_aggregation_threshold: float, n_to_show: int, **kwargs):
        super().__init__(**kwargs)
        self.segment_by = segment_by
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_features = n_top_features
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.loss_per_sample = loss_per_sample
        self.alternative_scorer = alternative_scorer if alternative_scorer else None
        self.categorical_aggregation_threshold = categorical_aggregation_threshold

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        text_data = context.get_data_by_kind(dataset_kind)
        text_data = text_data.sample(self.n_samples, random_state=context.random_state)

        if self.segment_by == 'metadata':
            context.assert_metadata(text_data=text_data)
            features = select_from_dataframe(text_data.metadata, self.columns, self.ignore_columns)
            cat_features = [col for col in features.columns if text_data.metadata_types[col] == 'categorical']
            features_name = 'metadata'

        elif self.segment_by == 'properties':
            context.assert_properties(text_data=text_data)
            features = select_from_dataframe(text_data.properties, self.columns, self.ignore_columns)
            cat_features = [col for col in features.columns if text_data.properties_types[col] == 'categorical']
            features_name = 'properties'
        else:
            raise DeepchecksProcessError(f'Unknown segment_by value: {self.segment_by}')

        self._warn_n_top_columns(features.shape[1])

        predictions = context.model.predict(text_data)

        if self.loss_per_sample is not None:
            loss_per_sample = self.loss_per_sample[list(text_data.index)]
            proba_values = None
        elif not hasattr(context.model, 'predict_proba'):
            raise DeepchecksNotSupportedError('Predicted probabilities not supplied. The weak segment checks relies'
                                              ' on log loss error that requires predicted probabilities, rather'
                                              ' than only predicted classes.')
        else:
            proba_values = np.asarray(context.model.predict_proba(text_data))
            loss_per_sample = [log_loss([y_true], [y_proba], labels=sorted(context.model_classes)) for y_true, y_proba
                               in zip(list(text_data.label), proba_values)]

        if features.shape[1] < 2:
            raise DeepchecksNotSupportedError('Check requires meta data to have at least two columns in order to run.')
        # label is not used in the check, just here to avoid errors
        dataset = Dataset(features, label=pd.Series(text_data.label), cat_features=cat_features)
        encoded_dataset = self._target_encode_categorical_features_fill_na(dataset, list(np.unique(text_data.label)))

        dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=np.asarray(predictions),
                                  y_proba_test=proba_values, validate_data_on_predict=False)
        scorer = context.get_single_scorer(self.alternative_scorer)
        weak_segments = self._weak_segments_search(dummy_model, encoded_dataset, np.asarray(encoded_dataset.features),
                                                   loss_per_sample, scorer)
        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         f'segments. Try increasing n_samples or supply more {features_name}.')

        avg_score = round(scorer(dummy_model, encoded_dataset), 3)
        display = self._create_heatmap_display(dummy_model, encoded_dataset, weak_segments, avg_score,
                                               scorer) if context.with_display else []

        for idx, segment in weak_segments.copy().iterrows():
            for feature in ['Feature1', 'Feature2']:
                if segment[feature] in encoded_dataset.cat_features:
                    weak_segments.at[idx, f'{feature} range'] = \
                        self._format_partition_vec_for_display(segment[f'{feature} range'], segment[feature], None)[0]

        display_msg = 'Showcasing intersections of metadata columns with weakest detected segments.<br> The full ' \
                      'list of weak segments can be observed in the check result value. '
        return CheckResult({'weak_segments_list': weak_segments, 'avg_score': avg_score, 'scorer_name': scorer.name},
                           display=[display_msg, DisplayMap(display)])

    def _warn_n_top_columns(self, n_columns: int):
        """Warn if n_top_columns is smaller than the number of segmenting features (metadata or properties)."""
        if self.n_top_features is not None and self.n_top_features < n_columns:
            if self.segment_by == 'metadata':
                features_name = 'metadata columns'
                n_top_columns_parameter = 'n_top_columns'
                columns_parameter = 'columns'
            else:
                features_name = 'properties'
                n_top_columns_parameter = 'n_top_properties'
                columns_parameter = 'properties'

            warnings.warn(
                f'Parameter {n_top_columns_parameter} is set to {self.n_top_features} to avoid long computation time. '
                f'This means that the check will run on the first {self.n_top_features} {features_name}. '
                f'If you want to run on all {features_name}, set {n_top_columns_parameter} to None. '
                f'Alternatively, you can set parameter {columns_parameter} to a list of the specific {features_name} '
                f'you want to run on.', UserWarning)


class PropertySegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the text properties - which are features extracted from the text, such as "language" and
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
    alternative_scorer : Tuple[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    loss_per_sample: Union[np.array, pd.Series, None], default: None
        Loss per sample used to detect relevant weak segments. If pd.Series the indexes should be similar to those in
        the dataset object provide, if np.array the order should be based on the index order of the dataset object and
        if None the check calculates loss per sample by via log loss for classification and MSE for regression.
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
                 alternative_scorer: Dict[str, Callable] = None,
                 loss_per_sample: Union[np.ndarray, pd.Series, None] = None,
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
                         loss_per_sample=loss_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         **kwargs)


class MetadataSegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the metadata - which is data that is not part of the text, but is related to it,
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
    alternative_scorer : Tuple[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    loss_per_sample: Union[np.array, pd.Series, None], default: None
        Loss per sample used to detect relevant weak segments. If pd.Series the indexes should be similar to those in
        the dataset object provide, if np.array the order should be based on the index order of the dataset object and
        if None the check calculates loss per sample by via log loss for classification and MSE for regression.
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
                 alternative_scorer: Dict[str, Callable] = None,
                 loss_per_sample: Union[np.ndarray, pd.Series, None] = None,
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
                         loss_per_sample=loss_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         **kwargs)
