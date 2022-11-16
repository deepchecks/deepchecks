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
import torch
from hamcrest import assert_that, close_to, calling, raises, has_items
from deepchecks.vision.checks import WeakSegmentsPerformance
from deepchecks.core.errors import DeepchecksProcessError
from tests.base.utils import equal_condition_result


def test_detection_defaults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.691, 0.001))


def test_detection_condition(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    check = WeakSegmentsPerformance().add_condition_segments_relative_performance_greater_than(0.5)

    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    condition_result = result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The relative performance of weakest segment is greater than 50% of average model performance.',
            details='Found a segment with average_loss score of 0.511 in comparison to an average score of 0.691 in '
                    'sampled data.')
    ))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device, n_samples=1000)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.067, 0.001))


def test_segmentation_defaults(segmentation_coco_test_visiondata, trained_segmentation_deeplabv3_mobilenet_model, device):
    check = WeakSegmentsPerformance()

    assert_that(calling(check.run).with_args(
        segmentation_coco_test_visiondata, trained_segmentation_deeplabv3_mobilenet_model, device),
        raises(DeepchecksProcessError))
