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
import torch
from ignite.metrics import Metric

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.metrics import DeepcheckScorer
from deepchecks.utils.performance.error_model import error_model_display

from deepchecks.vision import TrainTestCheck, Context
from deepchecks.vision.checks.distribution.image_property_drift import ImageProperty
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.iou_utils import compute_class_ious

__all__ = ['ModelErrorAnalysis']

from deepchecks.vision.utils import ImageFormatter


class ModelErrorAnalysis(TrainTestCheck):
    """TODO
    """

    def __init__(self,
                 image_properties: t.Optional[t.List[ImageProperty]] = None,
                 alternative_metrics: t.Dict[str, Metric] = None,
                 min_error_model_score: float = 0.5,
                 min_segment_size: float = 0.05,
                 max_properties_to_show: int = 20,
                 min_property_contribution: float = 0.15,
                 n_display_samples: int = 5_000,
                 random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.min_error_model_score = min_error_model_score
        self.min_segment_size = min_segment_size
        self.max_properties_to_show = max_properties_to_show
        self.min_property_contribution = min_property_contribution
        self.n_display_samples = n_display_samples
        # todo: get metric type but it's not metric
        self.alternative_metrics = alternative_metrics

        if image_properties is None:
            self.image_properties = ImageFormatter.IMAGE_PROPERTIES
        else:
            if len(image_properties) == 0:
                raise DeepchecksValueError('image_properties list cannot be empty')

            received_properties = {p for p in image_properties if isinstance(p, str)}
            unknown_properties = received_properties.difference(ImageFormatter.IMAGE_PROPERTIES)

            if len(unknown_properties) > 0:
                raise DeepchecksValueError(
                    'receivedd list of unknown image properties '
                    f'- {sorted(unknown_properties)}'
                )

            self.image_properties = image_properties

    def initialize_run(self, context: Context):
        """Initialize run."""
        self.train_properties = defaultdict(list)
        self.test_properties = defaultdict(list)
        self.train_scores = []
        self.test_scores = []

    def update(self, context: Context, batch: t.Any, dataset_kind):
        """Update."""
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
                # TODO: if it is a lambda it will have a name - <lambda>, that is a problem/
                properties[image_property.__name__].extend(image_property(images))
            else:
                raise DeepchecksValueError(
                    'Do not know how to work with image'
                    f'property of type - {type(image_property).__name__}'
                )

        if dataset.task_type == TaskType.CLASSIFICATION:
            def scoring_func(predictions, labels):
                import numpy as np
                y_pred = predictions.detach().numpy()
                y_true = labels.detach().numpy()
                return - (np.tile(y_true.reshape((-1, 1)), (1, y_pred.shape[1])) *
                          np.log(y_pred + np.finfo(float).eps)).sum(axis=1)

        elif dataset.task_type == TaskType.OBJECT_DETECTION:
            # TODO: conintue fixing this, ious are broken atm
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
        """Compute the metric result using the ignite metrics compute method and create display."""
        # build dataframe of properties and scores
        train_property_df = pd.DataFrame(self.train_properties).dropna(axis=1, how='all')
        test_property_df = pd.DataFrame(self.test_properties)[train_property_df.columns]

        from deepchecks.utils.performance.error_model import error_model_score

        error_fi, error_model_predicted, error_model = error_model_score(train_property_df,
                                                                        self.train_scores,
                                                                        test_property_df,
                                                                        self.test_scores,
                                                                        train_property_df.columns.to_list(),
                                                                        [],
                                                                        min_error_model_score=self.min_error_model_score,
                                                                        random_state=self.random_state)

        display, value = error_model_display(error_fi,
                                      test_property_df,
                                      DeepcheckScorer('accuracy', 'Accuracy'),
                                      self.max_properties_to_show,
                                      self.min_property_contribution,
                                      self.n_display_samples,
                                      error_model_predicted,
                                      [],
                                      self.min_segment_size,
                                      error_model,
                                      self.random_state)

        headnote = f"""<span>
            The following graphs show the distribution of error for top properties that are most useful for distinguishing
            high error samples from low error samples. Top properties are calculated using `feature_importances_`.
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)
