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
"""Module containing class performance check."""
import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.performance.error_model import error_model_display, error_model_score, \
    per_sample_binary_cross_entropy, error_model_display_dataframe

from deepchecks.vision import TrainTestCheck, Context
from deepchecks.vision.checks.distribution.image_property_drift import ImageProperty
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.iou_utils import compute_class_ious

__all__ = ['ModelErrorAnalysis']

from deepchecks.vision.utils import ImageFormatter


class ModelErrorAnalysis(TrainTestCheck):
    """Find the properties that best split the data into segments of high and low model error.

    The check trains a regression model to predict the error of the user's model. Then, the properties scoring the highest
    feature importance for the error regression model are selected and the distribution of the error vs the feature
    values is plotted. The check results are shown only if the error regression model manages to predict the error
    well enough.

    Parameters
    ----------
    image_properties : Optional[List[ImageProperty]] , default None
        An optional dictionary of properties to extract from na image. If none given, using default properties.
    max_properties_to_show : int , default: 3
        maximal number of properties to show error distribution for.
    min_property_contribution : float , default: 0.15
        minimum feature importance of a property to the error regression model
        in order to show the property.
    min_error_model_score : float , default: 0.5
        minimum r^2 score of the error regression model for displaying the check.
    min_segment_size : float , default: 0.05
        minimal fraction of data that can comprise a weak segment.
    n_display_samples : int , default: 5_000
        number of samples to display in scatter plot.
    random_seed : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 image_properties: t.Optional[t.List[ImageProperty]] = None,
                 min_error_model_score: float = 0.5,
                 min_segment_size: float = 0.05,
                 max_properties_to_show: int = 20,
                 min_property_contribution: float = 0.15,
                 n_display_samples: int = 5_000,
                 random_seed: int = 42):
        super().__init__()
        self.random_state = random_seed
        self.min_error_model_score = min_error_model_score
        self.min_segment_size = min_segment_size
        self.max_properties_to_show = max_properties_to_show
        self.min_property_contribution = min_property_contribution
        self.n_display_samples = n_display_samples

        if image_properties is None:
            self.image_properties = ImageFormatter.IMAGE_PROPERTIES
        else:
            if len(image_properties) == 0:
                raise DeepchecksValueError('image_properties list cannot be empty')

            received_properties = {p for p in image_properties if isinstance(p, str)}
            unknown_properties = received_properties.difference(ImageFormatter.IMAGE_PROPERTIES)

            if len(unknown_properties) > 0:
                raise DeepchecksValueError(
                    'received list of unknown image properties '
                    f'- {sorted(unknown_properties)}'
                )

            self.image_properties = image_properties

    def initialize_run(self, context: Context):
        """Initialize property and score lists."""
        self.train_properties = defaultdict(list)
        self.test_properties = defaultdict(list)
        self.train_scores = []
        self.test_scores = []

    def update(self, context: Context, batch: t.Any, dataset_kind):
        """Accumulate property data of images and scores."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self.train_properties
            scores = self.train_scores
        elif dataset_kind == DatasetKind.TEST:
            dataset = context.test
            properties = self.test_properties
            scores = self.test_scores
        else:
            raise RuntimeError(
                'Internal Error! Part of code that must '
                'be unreacheable was reached.'
            )

        images = dataset.image_formatter(batch)
        predictions = context.infer(batch)
        labels = dataset.label_formatter(batch)

        for image_property in self.image_properties:
            if isinstance(image_property, str):
                properties[image_property].extend(
                    getattr(dataset.image_formatter, image_property)(images)
                )
            elif callable(image_property):
                properties[image_property.__name__].extend(image_property(images))
            else:
                raise DeepchecksValueError(
                    'Do not know how to work with image'
                    f'property of type - {type(image_property).__name__}'
                )

        if dataset.task_type == TaskType.CLASSIFICATION:
            def scoring_func(predictions, labels):
                return per_sample_binary_cross_entropy(labels.detach().numpy(), predictions.detach().numpy())

        elif dataset.task_type == TaskType.OBJECT_DETECTION:
            def scoring_func(predictions, labels):
                mean_ious = []
                for detected, ground_truth in zip(predictions, labels):
                    if len(ground_truth) == 0:
                        if len(detected) == 0:
                            mean_ious.append(1)
                        else:
                            mean_ious.append(0)
                        continue

                    ious = compute_class_ious(detected, ground_truth)
                    count = 0
                    sum_iou = 0
                    for cls, cls_ious in ious.items():
                        for detection in cls_ious:
                            if len(detection):
                                sum_iou += max(detection)
                                count += 1
                    if count:
                        mean_ious.append(sum_iou/count)
                    else:
                        mean_ious.append(0)

                return mean_ious

        # get score using scoring_function
        scores.extend(scoring_func(predictions, labels))

    def compute(self, context: Context) -> CheckResult:
        """Train a model on the properties and errors as labels to find properties that contribute to the error, then
        get segments of these properties to display a split of the effected

        Returns
        -------
        CheckResult:
            value: dictionary of details for each property segment that split the effect on the error of the model
            display: plots of results
        """
        # build dataframe of properties and scores
        train_property_df = pd.DataFrame(self.train_properties).dropna(axis=1, how='all')
        test_property_df = pd.DataFrame(self.test_properties)[train_property_df.columns]

        error_fi, error_model_predicted = \
            error_model_score(train_property_df,
                              self.train_scores,
                              test_property_df,
                              self.test_scores,
                              train_property_df.columns.to_list(),
                              [],
                              min_error_model_score=self.min_error_model_score,
                              random_state=self.random_state)

        display, value = error_model_display_dataframe(error_fi,
                                                       error_model_predicted,
                                                       test_property_df,
                                                       [],
                                                       self.max_properties_to_show,
                                                       self.min_property_contribution,
                                                       self.n_display_samples,
                                                       self.min_segment_size,
                                                       self.random_state)

        headnote = f"""<span>
            The following graphs show the distribution of error for top properties that are most useful for 
            distinguishing high error samples from low error samples.
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)
