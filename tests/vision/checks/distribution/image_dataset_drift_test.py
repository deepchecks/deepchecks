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
from hamcrest import assert_that, has_entries, close_to, equal_to

import numpy as np
from deepchecks.vision.checks import ImageDatasetDrift
from deepchecks.vision.datasets.detection.coco import COCOData
from tests.vision.vision_conftest import *


def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.11).astype(int)
    return img + addition_of_brightness


def pil_formatter(batch):
    return [np.array(img) for img in batch[0]]


def pil_drift_formatter(batch):
    return [add_brightness(np.array(img)) for img in batch[0]]


def test_no_drift_grayscale(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test, random_state=42, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.494, 0.001),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(1),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    }))


def test_drift_grayscale(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test, random_state=42, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.509, 0.001),
        'domain_classifier_drift_score': close_to(0.019, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(1),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    }))


def test_no_drift_rgb(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    train = COCOData(coco_train_dataloader)
    test = COCOData(coco_test_dataloader)

    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test, random_state=42, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.494, 0.001),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'brightness': equal_to(1),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': equal_to(0),
            'normalized_green_mean': equal_to(0),
            'normalized_blue_mean': equal_to(0),
        })
    }))


def test_with_drift_rgb(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    train = COCOData(coco_train_dataloader)
    test = COCOData(coco_test_dataloader)
    check = ImageDatasetDrift()

    # Act
    result = check.run(train, test, random_state=42, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.747, 0.001),
        'domain_classifier_drift_score': close_to(0.494, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'brightness': close_to(1, 0.01),
            'aspect_ratio': equal_to(0),
            'area': equal_to(0),
            'normalized_red_mean': close_to(0, 0.01),
            'normalized_green_mean': close_to(0, 0.01),
            'normalized_blue_mean': close_to(0, 0.01),
        })
    }))
