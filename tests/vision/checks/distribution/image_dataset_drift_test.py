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
"""Test functions of the VISION train test label drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling

import numpy as np
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData
from deepchecks.vision.checks import ImageDatasetDrift
from deepchecks.vision.utils import ImageFormatter, DetectionLabelFormatter
from tests.vision.vision_conftest import *


def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.11).astype(int)
    return img + addition_of_brightness


def pil_formatter(batch):
    return [np.array(img) for img in batch[0]]


def pil_drift_formatter(batch):
    return [add_brightness(np.array(img)) for img in batch[0]]


def test_no_drift_grayscale(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = ImageDatasetDrift()
    train.set_seed(42)
    test.set_seed(42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.484, 0.01),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(0),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    })
                )


def test_drift_grayscale(mnist_dataset_train, mnist_dataset_test):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = ImageDatasetDrift()
    train.set_seed(42)
    test.set_seed(42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.524, 0.01),
        'domain_classifier_drift_score': close_to(0.048, 0.01),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(1),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    })
                )


def test_no_drift_rgb(coco_train_dataloader, coco_test_dataloader):
    # Arrange
    train = VisionData(coco_train_dataloader, image_formatter=ImageFormatter(pil_formatter),
                       label_formatter=DetectionLabelFormatter())
    test = VisionData(coco_test_dataloader, image_formatter=ImageFormatter(pil_formatter),
                      label_formatter=DetectionLabelFormatter())

    train.set_seed(42)
    test.set_seed(42)
    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.29, 0.01),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(0),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    })
                )


def test_with_drift_rgb(coco_train_dataloader, coco_test_dataloader):
    # Arrange
    train = VisionData(coco_train_dataloader, image_formatter=ImageFormatter(pil_drift_formatter),
                       label_formatter=DetectionLabelFormatter())
    test = VisionData(coco_test_dataloader, image_formatter=ImageFormatter(pil_formatter),
                      label_formatter=DetectionLabelFormatter())

    train.set_seed(42)
    test.set_seed(42)
    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.572, 0.001),
        'domain_classifier_drift_score': close_to(0.144, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'brightness': close_to(1, 0.01),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': close_to(0, 0.01),
            'normalized_green_mean': close_to(0, 0.01),
            'normalized_blue_mean': close_to(0, 0.01),
        })
    })
                )
