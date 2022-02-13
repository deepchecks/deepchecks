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
    # Act
    check = RobustnessReport(prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    result = check.run(mnist_dataset_train, trained_mnist)
    # Assert
    print(result.value)
    assert_that(result.value, has_entries({
        'RandomBrightnessContrast': has_entries({
            'Precision': has_entries(score=close_to(0.98, 0.01), diff=close_to(0, 0.02)),
            'Recall': has_entries(score=close_to(0.98, 0.01), diff=close_to(0, 0.02))
        }),
        'ShiftScaleRotate': has_entries({
            'Precision': has_entries(score=close_to(0.78, 0.01), diff=close_to(-0.2, 0.02)),
            'Recall': has_entries(score=close_to(0.78, 0.01), diff=close_to(-0.2, 0.02))
        }),
    }))


def test_coco(coco_train_visiondata, trained_yolov5_object_detection):
    # Act
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = RobustnessReport(prediction_formatter=pred_formatter)
    result = check.run(coco_train_visiondata, trained_yolov5_object_detection)
    # Assert
    assert_that(result.value, has_entries({
        'RandomBrightnessContrast': has_entries({
            'mAP': has_entries(score=close_to(0.5, 0.01), diff=close_to(0, 0.02)),
        }),
        'ShiftScaleRotate': has_entries({
            'mAP': has_entries(score=close_to(0.22, 0.01), diff=close_to(-0.54, 0.02)),
        }),
        'HueSaturationValue': has_entries({
            'mAP': has_entries(score=close_to(0.46, 0.01), diff=close_to(-0.16, 0.02)),
        }),
        'RGBShift': has_entries({
            'mAP': has_entries(score=close_to(0.5, 0.01), diff=close_to(0, 0.02)),
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
    check = RobustnessReport(prediction_formatter=pred_formatter)
    msg = r'Found that labels have not been affected by adding augmentation to field "transforms". This might be ' \
          r'a problem with the implementation of `Dataset.__getitem__`. label value: .*'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection),
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
    check = RobustnessReport(prediction_formatter=pred_formatter)
    msg = r'Found that images have not been affected by adding augmentation to field "transforms". This might be a ' \
          r'problem with the implementation of Dataset.__getitem__'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection),
                raises(DeepchecksValueError, msg))


def test_condition_degradation_not_greater_than_pass(mnist_dataset_train, trained_mnist):
    # Arrange
    check = RobustnessReport(prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    check.add_condition_degradation_not_greater_than(0.4)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist)
    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Metrics degrade by not more than 40%'
    ))


def test_condition_degradation_not_greater_than_fail(mnist_dataset_train, trained_mnist):
    # Arrange
    check = RobustnessReport(prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1)))
    check.add_condition_degradation_not_greater_than(0.01)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist)
    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name='Metrics degrade by not more than 1%',
        details='Augmentations not passing: {\'ShiftScaleRotate\'}'
    ))