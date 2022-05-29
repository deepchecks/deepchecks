# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
#
import copy
import numpy as np

import torch
from hamcrest import (assert_that, calling, close_to, equal_to, has_entries, has_items, has_length,
                      has_properties, has_property, instance_of, is_, raises)
from deepchecks.core.check_result import CheckResult

from deepchecks.vision.base_checks import SingleDatasetCheck
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.checks.model_evaluation.class_performance import ClassPerformance
from deepchecks.vision.checks.model_evaluation.image_segment_performance import ImageSegmentPerformance
from deepchecks.vision.checks.model_evaluation.train_test_prediction_drift import TrainTestPredictionDrift
from deepchecks.vision.context import Context
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.vision_data import VisionData
from tests.base.utils import equal_condition_result


class _StaticPred(SingleDatasetCheck):
    def initialize_run(self, context: Context, dataset_kind):
        self._pred_index = {}

    def update(self, context: Context, batch: Batch, dataset_kind):
        predictions = batch.predictions
        indexes = [batch.get_index_in_dataset(index) for index in range(len(predictions))]
        self._pred_index.update(dict(zip(indexes, predictions)))

    def compute(self, context: Context, dataset_kind) -> CheckResult:
        sorted_values = [v for _, v in sorted(self._pred_index.items(), key=lambda item: item[0])]
        if context.get_data_by_kind(dataset_kind).task_type == TaskType.CLASSIFICATION:
            sorted_values = torch.stack(sorted_values)
        return CheckResult(sorted_values)


def _create_static_predictions(train: VisionData, test: VisionData, model):
    static_preds = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_pred = _StaticPred().run(vision_data, model=model, n_samples=None).value
        else:
            static_pred = None
        static_preds.append(static_pred)
    train_preds, tests_preds = static_preds
    return train_preds, tests_preds


# copied from class_performance_test
def test_class_performance_mnist_largest(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    train_preds, tests_preds = _create_static_predictions(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist)
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test,
                       train_predictions=train_preds, test_predictions=tests_preds,
                       device=device, n_samples=None)
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(set(result.value['Class'])), equal_to(2))
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.977, 0.001))
    assert_that(first_row['Number of samples'], equal_to(6742))
    assert_that(first_row['Class'], equal_to(1))


# copied from image_segment_performance_test
def test_image_segment_performance_coco_and_condition(coco_train_visiondata, mock_trained_yolov5_object_detection):
    # Arrange
    train_preds, _ = _create_static_predictions(coco_train_visiondata, None, mock_trained_yolov5_object_detection)
    check = ImageSegmentPerformance().add_condition_score_from_mean_ratio_not_less_than(0.5) \
        .add_condition_score_from_mean_ratio_not_less_than(0.1)
    # Act
    result = check.run(coco_train_visiondata, train_predictions=train_preds)
    # Assert result
    assert_that(result.value, has_entries({
        'Mean Blue Relative Intensity': has_length(5),
        'Mean Green Relative Intensity': has_length(5),
        'Mean Red Relative Intensity': has_length(5),
        'Brightness': has_length(5),
        'Area': has_length(4),
        'Aspect Ratio': has_items(
            has_entries({
                'start': -np.inf, 'stop': 0.6671875, 'count': 12, 'display_range': '(-inf, 0.67)',
                'metrics': has_entries({'Average Precision': close_to(0.454, 0.001), 'Average Recall': close_to(0.45, 0.001)})
            }),
            has_entries({
                'start': 0.6671875, 'stop': 0.75, 'count': 11, 'display_range': '[0.67, 0.75)',
                'metrics': has_entries({'Average Precision': close_to(0.367, 0.001), 'Average Recall': close_to(0.4, 0.001)})
            }),
            has_entries({
                'start': 0.75, 'stop': close_to(1.102, 0.001), 'count': 28, 'display_range': '[0.75, 1.1)',
                'metrics': has_entries({'Average Precision': close_to(0.299, 0.001), 'Average Recall': close_to(0.333, 0.001)})
            }),
            has_entries({
                'start': close_to(1.102, 0.001), 'stop': np.inf, 'count': 13, 'display_range': '[1.1, inf)',
                'metrics': has_entries({'Average Precision': close_to(0.5, 0.001), 'Average Recall': close_to(0.549, 0.001)})
            }),
        )
    }))
    # Assert condition
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            name='No segment with ratio between score to mean less than 10%'
        ),
        equal_condition_result(
            is_pass=False,
            name='No segment with ratio between score to mean less than 50%',
            details="Properties with failed segments: Mean Green Relative Intensity: "
                    "{'Range': '[0.34, 0.366)', 'Metric': 'Average Precision', 'Ratio': 0.44}"
        )
    ))


# copied from train_test_prediction_drift_test
def test_train_test_prediction_with_drift_object_detection_change_max_cat(coco_train_visiondata, coco_test_visiondata,
                                                                          mock_trained_yolov5_object_detection, device):
    # Arrange
    train_preds, test_preds = _create_static_predictions(coco_train_visiondata,
                                                         coco_test_visiondata,
                                                         mock_trained_yolov5_object_detection)
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories_for_drift=100)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata,
                       train_predictions=train_preds, test_predictions=test_preds, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.48, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.085, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
