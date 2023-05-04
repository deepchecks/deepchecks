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
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.utils.abstracts.weak_segment_abstract import WeakSegmentAbstract
from deepchecks.utils.single_sample_metrics import per_sample_cross_entropy
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
    scorer: Optional[Callable], default: None
        a scorer (metric) to override the default scorer, callable that accepts (predictions, labels) and returns the
        score per sample.
    scorer_name: Optional[str], default: None
        The scorer name to display in the plots.
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numerical'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image properties, see the guide about :ref:`vision_properties_guide`.
    number_of_bins: int, default : 5
        Maximum number of bins to segment a single property into.
    number_of_samples_to_infer_bins : int, default : 1000
        Minimum number of samples to use to infer the bounds of the segments' bins
    n_top_properties: int , default: 10
        Number of features to use for segment search. Top columns are selected based on feature importance.
    n_to_show: int , default: 3
        number of segments with the weakest performance to show.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    {additional_check_init_params:2*indent}
    """

    def __init__(
            self,
            scorer: Optional[Callable] = None,
            scorer_name: Optional[str] = None,
            image_properties: List[Dict[str, Any]] = None,
            number_of_bins: int = 5,
            number_of_samples_to_infer_bins: int = 1000,
            n_top_properties: int = 10,
            n_to_show: int = 3,
            segment_minimum_size_ratio: float = 0.05,
            n_samples: Optional[int] = 10000,
            **kwargs
    ):
        super().__init__(**kwargs)
        if image_properties is not None and len(image_properties) < 2:
            raise DeepchecksNotSupportedError('Check requires at least two image properties in order to run.')
        self.image_properties = image_properties
        self.n_samples = n_samples
        self.n_top_features = n_top_properties
        self.scorer = scorer
        self.scorer_name = scorer_name
        self.number_of_bins = number_of_bins
        self.number_of_samples_to_infer_bins = number_of_samples_to_infer_bins
        self.n_to_show = n_to_show
        self.segment_minimum_size_ratio = segment_minimum_size_ratio
        self._properties_results = None
        self._sample_scores = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize the properties and sample scores states."""
        task_type = context.get_data_by_kind(dataset_kind).task_type
        if self.scorer is None:
            if task_type == TaskType.CLASSIFICATION:
                def scoring_func(predictions, labels):
                    return per_sample_cross_entropy(labels, predictions)

                self.scorer = scoring_func
                self.scorer_name = 'cross entropy'
            elif task_type == TaskType.OBJECT_DETECTION:
                self.scorer = per_sample_mean_iou
                self.scorer_name = 'mean IoU'
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                self.scorer = per_sample_dice
                self.scorer_name = 'Dice score'
            else:
                raise DeepchecksValueError(f'Default scorer is not defined for task type {task_type}.')
        self._properties_results = defaultdict(list)
        self._sample_scores = []

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Calculate the image properties and scores per image."""
        properties_results = batch.vision_properties(self.image_properties, PropertiesInputType.IMAGES)
        for prop_name, prop_value in properties_results.items():
            self._properties_results[prop_name].extend(prop_value)

        self._sample_scores.extend(self.scorer(batch.numpy_predictions, batch.numpy_labels))

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Find the segments with the worst performance."""
        results_dict = self._properties_results
        loss_per_sample_col = 'score'
        results_dict[loss_per_sample_col] = self._sample_scores
        results_df = pd.DataFrame(results_dict)
        properties_used = self.image_properties or default_image_properties
        cat_features = [p['name'] for p in properties_used if p['output_type'] == 'categorical']

        #  encoding categorical features based on the loss per sample (not smart but a way to gets the job done)
        encoded_dataset = self._target_encode_categorical_features_fill_na(results_df,
                                                                           loss_per_sample_col, cat_features)

        weak_segments = self._weak_segments_search(data=encoded_dataset.features_columns,
                                                   loss_per_sample=encoded_dataset.label_col,
                                                   scorer_name=self.scorer_name)
        if len(weak_segments) == 0:
            raise DeepchecksProcessError('WeakSegmentsPerformance was unable to train an error model to find weak '
                                         'segments. Try increasing n_samples or supply additional properties.')

        avg_score = round(results_df[loss_per_sample_col].mean(), 3)

        if context.with_display:
            display = self._create_heatmap_display(data=encoded_dataset.features_columns, weak_segments=weak_segments,
                                                   avg_score=avg_score, loss_per_sample=encoded_dataset.label_col,
                                                   scorer_name=self.scorer_name)
        else:
            display = []

        weak_segments = self._generate_check_value_display(weak_segments, cat_features)
        display_msg = 'Showcasing intersections of properties with weakest detected segments.<br> The full list of ' \
                      'weak segments can be observed in the check result value. '
        return CheckResult(
            {'weak_segments_list': weak_segments, 'avg_score': avg_score, 'scorer_name': self.scorer_name},
            display=[display_msg, DisplayMap(display)])
