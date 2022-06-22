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
import numpy as np
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.vision.checks import ImageDatasetDrift
from deepchecks.vision.datasets.detection.coco import COCOData
from tests.base.utils import equal_condition_result
from tests.vision.vision_conftest import *


def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.31).astype(int)
    return img + addition_of_brightness


def pil_drift_formatter(batch):
    return [add_brightness(np.array(img)) for img in batch[0]]


def test_no_drift_grayscale(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = ImageDatasetDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, random_state=42, device=device, n_samples=None)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.479, 0.001),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'Brightness': equal_to(0),
            'Aspect Ratio': equal_to(0),
            'Area': equal_to(0),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': equal_to(0),
            'Mean Green Relative Intensity': equal_to(0),
        })
    }))


def test_no_drift_grayscale_cramer(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = ImageDatasetDrift(categorical_drift_method='cramer_v')

    # Act
    result = check.run(train, test, random_state=42, device=device, n_samples=None)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.479, 0.001),
        'domain_classifier_drift_score': equal_to(0),
        'domain_classifier_feature_importance': has_entries({
            'Brightness': equal_to(0),
            'Aspect Ratio': equal_to(0),
            'Area': equal_to(0),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': equal_to(0),
            'Mean Green Relative Intensity': equal_to(0),
        })
    }))


def test_drift_grayscale(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = ImageDatasetDrift(categorical_drift_method='PSI', min_meaningful_drift_score=-1)

    # Act
    result = check.run(train, test, random_state=42, device=device, n_samples=None)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.5146, 0.001),
        'domain_classifier_drift_score': close_to(0.029, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'RMS Contrast': close_to(1, 0.001),
            'Brightness': close_to(0, 0.001),
            'Aspect Ratio': equal_to(0),
            'Area': equal_to(0),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Green Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': equal_to(0),
        })
    }))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_grayscale_without_display(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = ImageDatasetDrift(categorical_drift_method='PSI', min_meaningful_drift_score=-1)

    # Act
    result = check.run(train, test, random_state=42, device=device, n_samples=None, with_display=False)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.5146, 0.001),
        'domain_classifier_drift_score': close_to(0.029, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'RMS Contrast': close_to(1, 0.001),
            'Brightness': close_to(0, 0.001),
            'Aspect Ratio': equal_to(0),
            'Area': equal_to(0),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Green Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': equal_to(0),
        })
    }))
    assert_that(result.display, has_length(0))


def test_no_drift_rgb(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    train = COCOData(coco_train_dataloader)
    test = COCOData(coco_test_dataloader)
    check = ImageDatasetDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, random_state=42, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.623, 0.001),
        'domain_classifier_drift_score': close_to(0.247, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'RMS Contrast': equal_to(0),
            'Brightness': close_to(0, 0.01),
            'Aspect Ratio': close_to(0, 0.01),
            'Area': close_to(0, 0.001),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': close_to(0, 0.001),
            'Mean Green Relative Intensity': close_to(1, 0.001),
        })
    }))


def test_with_drift_rgb(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    class DriftCoco(COCOData):
        def batch_to_images(self, batch):
            return pil_drift_formatter(batch)

    train = DriftCoco(coco_train_dataloader)
    test = COCOData(coco_test_dataloader)

    check = ImageDatasetDrift(categorical_drift_method='PSI')
    # Act
    result = check.run(train, test, random_state=42, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.908, 0.001),
        'domain_classifier_drift_score': close_to(0.815, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'Brightness': close_to(1, 0.001),
            'Aspect Ratio': equal_to(0),
            'Area': equal_to(0),
            'Mean Red Relative Intensity': close_to(0, 0.01),
            'Mean Green Relative Intensity': close_to(0, 0.01),
            'Mean Blue Relative Intensity': close_to(0, 0.01),
        })
    }))


def test_condition_fail(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    class DriftCoco(COCOData):
        def batch_to_images(self, batch):
            return pil_drift_formatter(batch)

    train = DriftCoco(coco_train_dataloader)
    test = COCOData(coco_test_dataloader)
    check = ImageDatasetDrift().add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, random_state=42, device=device)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Drift score is less than 0.1',
        details=f'Drift score 0.816 is not less than 0.1',
        ))


def test_condition_pass(coco_train_dataloader, coco_test_dataloader, device):
    # Arrange
    test = COCOData(coco_test_dataloader)
    check = ImageDatasetDrift().add_condition_drift_score_less_than(0.3)

    # Act
    result = check.run(test, test, random_state=42, device=device)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name=f'Drift score is less than 0.3',
        details=f'Drift score 0 is less than 0.3',
        ))
