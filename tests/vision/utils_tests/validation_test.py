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
import os

from hamcrest import has_items, assert_that, has_length, close_to, raises

from deepchecks.vision.utils.validation import validate_extractors
from deepchecks.vision.detection_data import DetectionData
from deepchecks.utils.ipython import is_headless

FILE_NAME = 'deepchecks_formatted_image.jpg'

def test_mnist_validation(mnist_dataset_train, mock_trained_mnist):
    # Arrange
    validate_extractors(mnist_dataset_train, mock_trained_mnist)
    if is_headless():
        assert_that(os.path.exists(FILE_NAME), True)
        assert_that(os.path.isfile(FILE_NAME), True)


def test_mnist_validation_no_save(mnist_dataset_train, mock_trained_mnist):
    # Arrange
    validate_extractors(mnist_dataset_train, mock_trained_mnist, save_images=False)
    if is_headless():
        assert_that(os.path.exists(FILE_NAME), False)


def test_coco_validation(coco_test_visiondata, mock_trained_yolov5_object_detection):
    # Arrange
    validate_extractors(coco_test_visiondata, mock_trained_yolov5_object_detection)
    if is_headless():
        assert_that(os.path.exists(FILE_NAME), True)
        assert_that(os.path.isfile(FILE_NAME), True)


def test_coco_validation_no_save(coco_test_visiondata, mock_trained_yolov5_object_detection):
    # Arrange
    validate_extractors(coco_test_visiondata, mock_trained_yolov5_object_detection, save_images=False)
    if is_headless():
        assert_that(os.path.exists(FILE_NAME), False)
