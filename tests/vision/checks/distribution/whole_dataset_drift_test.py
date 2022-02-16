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
from deepchecks.vision.checks import WholeDatasetDrift
from deepchecks.vision.utils import ImageFormatter, DetectionLabelFormatter
from tests.vision.vision_conftest import *


def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.11).astype(int)
    return img + addition_of_brightness


def pil_formatter(batch):
    return [np.array(img) for img in batch]


def pil_drift_formatter(batch):
    return [add_brightness(np.array(img)) for img in batch]


def test_no_drift_grayscale(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = WholeDatasetDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.48, 0.01),
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
    check = WholeDatasetDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.51, 0.01),
        'domain_classifier_drift_score': close_to(0.017, 0.01),
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
    train = VisionData(coco_train_dataloader, image_transformer=ImageFormatter(pil_formatter),
                       label_transformer=DetectionLabelFormatter(lambda x: x))
    test = VisionData(coco_test_dataloader, image_transformer=ImageFormatter(pil_formatter),
                      label_transformer=DetectionLabelFormatter(lambda x: x))

    check = WholeDatasetDrift()

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
    train = VisionData(coco_train_dataloader, image_transformer=ImageFormatter(pil_drift_formatter),
                       label_transformer=DetectionLabelFormatter(lambda x: x))
    test = VisionData(coco_test_dataloader, image_transformer=ImageFormatter(pil_formatter),
                      label_transformer=DetectionLabelFormatter(lambda x: x))

    check = WholeDatasetDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.619, 0.001),
        'domain_classifier_drift_score': close_to(0.239, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'brightness': close_to(0.62, 0.01),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': close_to(0.06, 0.01),
            'normalized_green_mean': close_to(0.15, 0.01),
            'normalized_blue_mean': close_to(0.15, 0.01),
        })
    })
                )
