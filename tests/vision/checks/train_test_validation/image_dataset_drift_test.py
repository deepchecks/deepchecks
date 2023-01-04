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
from deepchecks.vision.datasets.detection.coco_torch import collate_without_model
from deepchecks.vision.utils.test_utils import replace_collate_fn_visiondata
from tests.base.utils import equal_condition_result


def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.31).astype(int)
    return img + addition_of_brightness


def pil_drift_formatter(images):
    return [add_brightness(np.array(img)) for img in images]


def test_no_drift_grayscale(mnist_visiondata_train):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_train
    check = ImageDatasetDrift(categorical_drift_method='PSI', n_samples=10000)

    # Act
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.254, 0.001),
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


def test_no_drift_grayscale_cramer(mnist_visiondata_train):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_train
    check = ImageDatasetDrift(categorical_drift_method='cramer_v', n_samples=10000)

    # Act
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.254, 0.001),
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


def test_drift_grayscale(mnist_drifted_datasets):
    # Arrange
    train, test = mnist_drifted_datasets
    check = ImageDatasetDrift(categorical_drift_method='PSI', min_meaningful_drift_score=-1, n_samples=None)

    # Act
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.525, 0.001),
        'domain_classifier_drift_score': close_to(0.051, 0.001),
        'domain_classifier_feature_importance': has_entries({'Brightness': close_to(0.852, 0.001)})}))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_grayscale_without_display(mnist_drifted_datasets):
    # Arrange
    train, test = mnist_drifted_datasets
    check = ImageDatasetDrift(categorical_drift_method='PSI', min_meaningful_drift_score=-1)

    # Act
    result = check.run(train, test, with_display=False)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.525, 0.001),
        'domain_classifier_drift_score': close_to(0.051, 0.001),
        'domain_classifier_feature_importance': has_entries({'Brightness': close_to(0.852, 0.001)})}))
    assert_that(result.display, has_length(0))


def test_no_drift_rgb(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = ImageDatasetDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than(0.3)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.277, 0.001),
        'domain_classifier_drift_score': close_to(0, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'RMS Contrast': equal_to(0),
            'Brightness': close_to(0, 0.01),
            'Aspect Ratio': close_to(0, 0.01),
            'Area': close_to(0, 0.001),
            'Mean Red Relative Intensity': equal_to(0),
            'Mean Blue Relative Intensity': close_to(0, 0.001),
            'Mean Green Relative Intensity': close_to(0, 0.001),
        })
    }))
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name=f'Drift score is less than 0.3',
        details=f'Drift score 0 is less than 0.3',
    ))


def test_with_drift_rgb(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    def collate_generate_drift(batch):
        images, labels = collate_without_model(batch)
        return {'images': pil_drift_formatter(images), 'labels': labels}

    drifted_train = replace_collate_fn_visiondata(coco_visiondata_train, collate_generate_drift)

    check = ImageDatasetDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()
    # Act
    result = check.run(drifted_train, coco_visiondata_test)
    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.977, 0.001),
        'domain_classifier_drift_score': close_to(0.955, 0.001),
        'domain_classifier_feature_importance': has_entries({
            'Brightness': close_to(0.967, 0.001),
        })
    }))
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Drift score is less than 0.1',
        details=f'Drift score 0.955 is not less than 0.1',
    ))
