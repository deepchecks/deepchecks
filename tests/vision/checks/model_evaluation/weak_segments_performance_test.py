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
from hamcrest import assert_that, close_to,calling, raises
from deepchecks.vision.checks import WeakSegmentsPerformance
from deepchecks.vision.utils.image_properties import brightness
from deepchecks.core.errors import DeepchecksProcessError


def test_detection_defaults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.691, 0.001))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.099, 0.001))


def test_segmentation_defaults(segmentation_coco_test_visiondata, trained_segmentation_deeplabv3_mobilenet_model, device):
    check = WeakSegmentsPerformance()

    assert_that(calling(check).with_args(
        segmentation_coco_test_visiondata, trained_segmentation_deeplabv3_mobilenet_model, device),
        raises(DeepchecksProcessError))
