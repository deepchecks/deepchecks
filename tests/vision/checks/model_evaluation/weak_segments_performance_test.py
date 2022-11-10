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

from hamcrest import assert_that, close_to, has_length
from deepchecks.vision.checks import WeakSegmentsPerformance


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