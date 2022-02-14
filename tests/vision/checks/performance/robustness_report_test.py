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
import types

import albumentations
import numpy as np

from tests.checks.utils import equal_condition_result
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks.performance.robustness_report import RobustnessReport
from deepchecks.vision.utils.classification_formatters import ClassificationPredictionFormatter
from deepchecks.vision.utils import DetectionPredictionFormatter
from deepchecks.vision.datasets.detection.coco import yolo_prediction_formatter
from PIL import Image

import torch.nn as nn

from hamcrest import assert_that, has_entries, close_to, calling, raises
from tests.vision.vision_conftest import *


def test_mnist(mnist_dataset_train, trained_mnist):
    # Arrange
    # Create augmentations without randomness to get fixed metrics results
    augmentations = [
        albumentations.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.2, 0.2), p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=(0.1, 0.1), scale_limit=(0.1, 0.1), rotate_limit=(10, 10), p=1.0),
    ]
    check = RobustnessReport(augmentations=augmentations)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    # Assert
    assert_that(result.value, has_entries({
        'RandomBrightnessContrast': has_entries({
            'Precision': has_entries(score=close_to(0.98, 0.1), diff=close_to(0, 0.15)),
            'Recall': has_entries(score=close_to(0.98, 0.1), diff=close_to(0, 0.15))
        }),
        'ShiftScaleRotate': has_entries({
            'Precision': has_entries(score=close_to(0.40, 0.1), diff=close_to(-0.59, 0.15)),
            'Recall': has_entries(score=close_to(0.38, 0.1), diff=close_to(-0.6, 0.15))
        }),
    }))


def test_coco(coco_train_visiondata, trained_yolov5_object_detection):
    # Arrange
    # Create augmentations without randomness to get fixed metrics results
    augmentations = [
        albumentations.RGBShift(r_shift_limit=(15, 15), g_shift_limit=(15, 15), b_shift_limit=(15, 15), p=1.0)
    ]
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = RobustnessReport(augmentations=augmentations)
    # Act
    result = check.run(coco_train_visiondata, trained_yolov5_object_detection, prediction_formatter=pred_formatter)
    # Assert
    assert_that(result.value, has_entries({
        'RGBShift': has_entries({
            'mAP': has_entries(score=close_to(0.5, 0.1), diff=close_to(0, 0.1)),
        }),
    }))


def test_dataset_not_augmenting_labels(coco_train_visiondata, trained_yolov5_object_detection):
    # Arrange
    vision_data = coco_train_visiondata.copy()
    dataset = vision_data.get_data_loader().dataset

    def new_apply(self, img, bboxes):
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img), bboxes=bboxes)
            img = Image.fromarray(transformed['image'])
        return img, bboxes
    dataset.apply_transform = types.MethodType(new_apply, dataset)

    vision_data.add_augmentation(albumentations.ShiftScaleRotate(p=1))
    # Act & Assert
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = RobustnessReport()
    msg = r'Found that labels have not been affected by adding augmentation to field "transforms". This might be ' \
          r'a problem with the implementation of `Dataset.__getitem__`. label value: .*'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection,
                                             prediction_formatter=pred_formatter),
                raises(DeepchecksValueError, msg))


def test_dataset_not_augmenting_data(coco_train_visiondata, trained_yolov5_object_detection):
    # Arrange
    vision_data = coco_train_visiondata.copy()
    dataset = vision_data.get_data_loader().dataset

    def new_apply(self, img, bboxes):
        return img, bboxes
    dataset.apply_transform = types.MethodType(new_apply, dataset)
    vision_data.add_augmentation(albumentations.ShiftScaleRotate(p=1))

    # Act & Assert
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = RobustnessReport()
    msg = r'Found that images have not been affected by adding augmentation to field "transforms". This might be a ' \
          r'problem with the implementation of Dataset.__getitem__'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection,
                                             prediction_formatter=pred_formatter),
                raises(DeepchecksValueError, msg))


def test_condition_degradation_not_greater_than_pass(mnist_dataset_train, trained_mnist):
    # Arrange
    check = RobustnessReport()
    check.add_condition_degradation_not_greater_than(0.4)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Metrics degrade by not more than 40%'
    ))


def test_condition_degradation_not_greater_than_fail(mnist_dataset_train, trained_mnist):
    # Arrange
    check = RobustnessReport()
    check.add_condition_degradation_not_greater_than(0.01)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name='Metrics degrade by not more than 1%',
        details='Augmentations not passing: {\'ShiftScaleRotate\'}'
    ))