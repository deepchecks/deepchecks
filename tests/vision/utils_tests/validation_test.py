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

from hamcrest import assert_that

from deepchecks.utils.ipython import is_headless
from deepchecks.vision.utils.validation import validate_extractors

FILE_NAME = 'deepchecks_formatted_image.jpg'

def test_mnist_validation(mnist_dataset_train, mock_trained_mnist):
    if is_headless():
        validate_extractors(mnist_dataset_train, mock_trained_mnist)
        assert_that(os.path.exists(FILE_NAME), True)
        assert_that(os.path.isfile(FILE_NAME), True)


def test_mnist_validation_no_save(mnist_dataset_train, mock_trained_mnist):
    if is_headless():
        validate_extractors(mnist_dataset_train, mock_trained_mnist, save_images=False)
        assert_that(os.path.exists(FILE_NAME), False)


def test_mnist_validation_new_loc_save(mnist_dataset_train, mock_trained_mnist):
    if is_headless():
        validate_extractors(mnist_dataset_train, mock_trained_mnist, image_save_location='/tmp')
        assert_that(os.path.exists('/tmp/' + FILE_NAME), False)


def test_coco_validation(coco_test_visiondata, mock_trained_yolov5_object_detection):
    if is_headless():
        validate_extractors(coco_test_visiondata, mock_trained_yolov5_object_detection)
        assert_that(os.path.exists(FILE_NAME), True)
        assert_that(os.path.isfile(FILE_NAME), True)


def test_coco_validation_no_save(coco_test_visiondata, mock_trained_yolov5_object_detection):
    if is_headless():
        validate_extractors(coco_test_visiondata, mock_trained_yolov5_object_detection, save_images=False)
        assert_that(os.path.exists(FILE_NAME), False)
