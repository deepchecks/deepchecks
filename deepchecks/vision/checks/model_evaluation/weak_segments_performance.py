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
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.single_sample_metrics import calculate_neg_cross_entropy_per_sample
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import SingleDatasetCheck
from deepchecks.vision.context import Context
from deepchecks.vision.metrics_utils.iou_utils import per_sample_mean_iou
from deepchecks.vision.metrics_utils.semantic_segmentation_metrics import per_sample_dice
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper


@docstrings
class WeakSegmentsPerformance(SingleDatasetCheck, WeakSegmentAbstract):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the image properties - characteristics of each image such as the contrast.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numerical'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image properties, see the guide about :ref:`vision__properties_guide`.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    max_categories_weak_segment: Optional[int] , default: None
        Maximum number of categories that can be included in a weak segment per categorical feature.
        If None, the number of categories is not limited.
    n_samples : Optional[int] , default: 10_000
        number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        For each categorical property, categories with frequency below threshold will be merged into "Other" category.
    multiple_segments_per_property : bool , default: True
        If True, will allow the same property to be a segmenting feature in multiple segments,
        otherwise each property can appear in one segment at most.
    {additional_check_init_params:2*indent}
    """

    def __init__(
            self,
            image_properties: List[Dict[str, Any]] = None,
            n_to_show: int = 3,
            segment_minimum_size_ratio: float = 0.05,
            max_categories_weak_segment: Optional[int] = None,
            n_samples: Optional[int] = 10000,
            categorical_aggregation_threshold: float = 0.05,
            multiple_segments_per_property: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if image_properties is not None and len(image_properties) < 2:
            raise DeepchecksNotSupportedError('Check requires at least two image properties in order to run.')
        self.image_properties = image_properties
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self.max_categories_weak_segment = max_categories_weak_segment
        self.categorical_aggregation_threshold = categorical_aggregation_threshold
        self.multiple_segments_per_property = multiple_segments_per_property
        self._properties_results = None
        self._sample_scores = None
        self._scorer_name = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize the properties and sample scores states."""
        task_type = context.get_data_by_kind(dataset_kind).task_type
        if task_type == TaskType.CLASSIFICATION:
            self._scorer_name = 'Cross Entropy'
        elif task_type == TaskType.OBJECT_DETECTION:
            self._scorer_name = 'Mean IoU'
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            self._scorer_name = 'Dice'
        else:
            raise DeepchecksValueError(f'Default scorer is not defined for task type {task_type}.')

        self._properties_results = defaultdict(list)
        self._sample_scores = []

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Calculate the image properties and scores per image."""
        properties_results = batch.vision_properties(self.image_properties, PropertiesInputType.IMAGES)
        for prop_name, prop_value in properties_results.items():
            self._properties_results[prop_name].extend(prop_value)

        if self._scorer_name == 'Cross Entropy':
            batch_per_sample_score = calculate_neg_cross_entropy_per_sample(np.asarray(batch.numpy_labels),
                                                                            np.asarray(batch.numpy_predictions))
        elif self._scorer_name == 'Mean IoU':
            batch_per_sample_score = per_sample_mean_iou(batch.numpy_predictions, batch.numpy_labels)
        elif self._scorer_name == 'Dice':
            batch_per_sample_score = per_sample_dice(batch.numpy_predictions, batch.numpy_labels)

        self._sample_scores.extend(list(batch_per_sample_score))

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Find the segments with the worst performance."""
        results_df = pd.DataFrame(self._properties_results)
        score_per_sample_col = pd.Series(self._sample_scores, index=results_df.index)
        properties_used = self.image_properties or default_image_properties
        cat_features = [p['name'] for p in properties_used if p['output_type'] == 'categorical']

        # Encoding categorical properties based on the loss per sample (not smart but a way to gets the job done)
        encoded_dataset = self._target_encode_categorical_features_fill_na(results_df,
                                                                           score_per_sample_col,
                                                                           cat_features,
                                                                           is_cat_label=False)

        weak_segments = self._weak_segments_search(data=encoded_dataset.features_columns,
                                                   score_per_sample=score_per_sample_col,
                                                   scorer_name=self._scorer_name,
                                                   multiple_segments_per_feature=self.multiple_segments_per_property)
        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         'segments. Try increasing n_samples or supply additional properties.')
        avg_score = round(score_per_sample_col.mean(), 3)

        if context.with_display:
            display = self._create_heatmap_display(data=encoded_dataset.features_columns, weak_segments=weak_segments,
                                                   avg_score=avg_score, score_per_sample=score_per_sample_col,
                                                   scorer_name=self._scorer_name)
        else:
            display = []

        check_result_value = self._generate_check_result_value(weak_segments, cat_features, avg_score)
        display_msg = 'Showcasing intersections of properties with weakest detected segments.<br> The full list of ' \
                      'weak segments can be observed in the check result value. '
        return CheckResult(value=check_result_value,
                           display=[display_msg, DisplayMap(display)])
