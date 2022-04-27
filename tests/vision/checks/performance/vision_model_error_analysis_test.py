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
"""Test functions of the VISION model error analysis."""

from hamcrest import assert_that, equal_to, instance_of

from deepchecks import CheckFailure
from deepchecks.vision.checks import ModelErrorAnalysis


def test_classification(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = ModelErrorAnalysis(min_error_model_score=0)
    train, test = mnist_dataset_train, mnist_dataset_train

    # Act
    result = check.run(train, test, mock_trained_mnist,
                       device=device, n_samples=None)
    # Assert
    assert_that(len(result.value['feature_segments']), equal_to(2))
    assert_that(result.value['feature_segments']['Brightness']['segment1']['n_samples'], equal_to(254))


def test_detection(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = ModelErrorAnalysis(min_error_model_score=-1)

    # Act
    result = check.run(coco_train_visiondata,
                       coco_test_visiondata,
                       mock_trained_yolov5_object_detection,
                       device=device)
    # Assert
    assert_that(len(result.value['feature_segments']), equal_to(3))
    assert_that(result.value['feature_segments']['Mean Green Relative Intensity']['segment1']['n_samples'],
                equal_to(21))


def test_classification_not_interesting(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = ModelErrorAnalysis(min_error_model_score=1)
    train, test = mnist_dataset_train, mnist_dataset_train

    # Assert
    assert_that(check.run(
        train, test, mock_trained_mnist,
        device=device), instance_of(CheckFailure))
