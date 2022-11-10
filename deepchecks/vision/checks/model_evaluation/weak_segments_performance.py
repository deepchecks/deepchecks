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

import numpy as np
import pandas as pd
from typing import Dict, Callable, List, Any, Union, Optional

from deepchecks.tabular.context import _DummyModel
from deepchecks.vision import Context, SingleDatasetCheck, Batch
from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from ignite.metrics import Metric

from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.utils.single_sample_metrics import per_sample_cross_entropy
from deepchecks.vision.metrics_utils.iou_utils import per_sample_mean_iou
from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError, DeepchecksProcessError
from deepchecks.utils.performance.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.tabular import Dataset
from deepchecks.core.check_result import DisplayMap


class WeakSegmentsPerformance(SingleDatasetCheck, WeakSegmentAbstract):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.
    """

    def __init__(
        self,
        scorer: Optional[Callable] = None,
        image_properties: List[Dict[str, Any]] = None,
        number_of_bins: int = 5,
        number_of_samples_to_infer_bins: int = 1000,
        n_to_show: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_properties = image_properties if image_properties else default_image_properties
        if len(self.image_properties) < 2:
            raise DeepchecksNotSupportedError('Check requires at least two image properties in order to run.')
        self.scorer = scorer
        self.number_of_bins = number_of_bins
        self.number_of_samples_to_infer_bins = number_of_samples_to_infer_bins
        self.n_to_show = n_to_show
        self._properties_results = None
        self._sample_scores = None
        self._predictions = None
        self._labels = None
        self._dummy_scorer = AvgLossScorer()

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        task_type = context.get_data_by_kind(dataset_kind).task_type
        if self.scorer is None:
            if task_type == TaskType.CLASSIFICATION:
                def scoring_func(predictions, labels):
                    return per_sample_cross_entropy(labels, predictions)
                self.scorer = scoring_func
            elif task_type == TaskType.OBJECT_DETECTION:
                self.scorer = per_sample_mean_iou
            else:
                raise DeepchecksValueError(f'Default scorer is not defined for task type {task_type}.')
        self._properties_results = defaultdict(list)
        self._sample_scores = []
        self._predictions = []
        self._labels = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate the image properties and scores per image."""
        properties_results = batch.vision_properties(self.image_properties, PropertiesInputType.IMAGES)
        for prop_name, prop_value in properties_results.items():
            self._properties_results[prop_name].extend(prop_value)

        predictions = [tens.detach() for tens in batch.predictions]
        labels = [tens.detach() for tens in batch.labels]
        self._sample_scores.extend(self.scorer(predictions, labels))
        self._predictions.extend(predictions)
        self._labels.extend(labels)

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Find the segments with the worst performance."""
        results_dict = self._properties_results
        results_dict['score'] = self._sample_scores
        results_dict['target'] = self._labels
        results_df = pd.DataFrame(results_dict)

        cat_features = [p for p in self.image_properties if p['output_type'] == 'categorical']
        num_features = [p for p in self.image_properties if p['output_type'] == 'numerical']
        all_features = cat_features + num_features

        dataset = Dataset(results_df, cat_features=cat_features, features=all_features)

        encoded_dataset = self._target_encode_categorical_features_fill_na(dataset, context.observed_classes)
        dummy_model = _DummyModel(test=encoded_dataset, y_pred_test=self._predictions, y_proba_test=self._predictions,
                                  validate_data_on_predict=False)
        feature_rank = np.asarray(range(len(all_features)))

        weak_segments = self._weak_segments_search(dummy_model, encoded_dataset, feature_rank,
                                                   self._sample_scores, self.scorer)
        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         'segments. Try increasing n_samples or supply additional properties.')

        avg_score = round(results_df['score'].mean(), 3)

        display = self._create_heatmap_display(dummy_model, encoded_dataset, weak_segments, avg_score,
                                               self._dummy_scorer) if context.with_display else []

        for idx, segment in weak_segments.copy().iterrows():
            for feature in ['Feature1', 'Feature2']:
                if segment[feature] in encoded_dataset.cat_features:
                    weak_segments[f'{feature} range'][idx] = \
                        self._format_partition_vec_for_display(segment[f'{feature} range'], segment[feature], None)[0]

        display_msg = 'Showcasing intersections of features with weakest detected segments.<br> The full list of ' \
                      'weak segments can be observed in the check result value. '
        return CheckResult({'weak_segments_list': weak_segments, 'avg_score': avg_score, 'scorer_name': 'average_loss'},
                           display=[display_msg, DisplayMap(display)])


class AvgLossScorer:
    """Patch for using the tabular methods from a dataframe of pre-calculated loss."""
    def __init__(self):
        self.name = 'average_loss'

    def run_on_data_and_label(self, model, data: pd.DataFrame, loss_col: pd.Series):
        """Patch for using the tabular methods from a dataframe of pre-calculated loss."""
        return loss_col.mean()
