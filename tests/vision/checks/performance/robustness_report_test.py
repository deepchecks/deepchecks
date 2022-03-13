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
from PIL import Image
from hamcrest import assert_that, has_entries, close_to, calling, raises, has_items

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks.performance.robustness_report import RobustnessReport
from deepchecks.vision.datasets.detection.coco import COCOData, CocoDataset
from tests.checks.utils import equal_condition_result


def test_mnist(mnist_dataset_train, trained_mnist, device):
    # Arrange
    # Create augmentations without randomness to get fixed metrics results
    augmentations = [
        albumentations.RandomBrightnessContrast(p=1.0),
        albumentations.ShiftScaleRotate(p=1.0),
    ]
    check = RobustnessReport(augmentations=augmentations)
    # Act
    result = check.run(mnist_dataset_train, trained_mnist, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Random Brightness Contrast': has_entries({
            'Precision': has_entries(score=close_to(0.984, 0.001), diff=close_to(-0.002, 0.001)),
            'Recall': has_entries(score=close_to(0.987, 0.001), diff=close_to(-0.0003, 0.001))
        }),
        'Shift Scale Rotate': has_entries({
            'Precision': has_entries(score=close_to(0.794, 0.001), diff=close_to(-0.194, 0.001)),
            'Recall': has_entries(score=close_to(0.776, 0.001), diff=close_to(-0.213, 0.001))
        }),
    }))


def test_coco_and_condition(coco_train_visiondata, trained_yolov5_object_detection, device):
    """Because of the large running time, instead of checking the conditions in separated tests, combining a few
    tests into one."""
    # Arrange
    # Create augmentations without randomness to get fixed metrics results
    augmentations = [
        albumentations.HueSaturationValue(p=1.0),
    ]
    check = RobustnessReport(augmentations=augmentations)

    check.add_condition_degradation_not_greater_than(0.5)
    check.add_condition_degradation_not_greater_than(0.01)

    # Act
    result = check.run(coco_train_visiondata, trained_yolov5_object_detection, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Hue Saturation Value': has_entries({
            'AP': has_entries(score=close_to(0.299, 0.001), diff=close_to(-0.079, 0.001)),
            'AR': has_entries(score=close_to(0.316, 0.001), diff=close_to(-0.136, 0.001))
        }),
    }))
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            name='Metrics degrade by not more than 50%'
        ),
        equal_condition_result(
            is_pass=False,
            name='Metrics degrade by not more than 1%',
            details='Augmentations not passing: {\'Hue Saturation Value\'}'
        )
    ))


def test_dataset_not_augmenting_labels(coco_train_visiondata: COCOData, trained_yolov5_object_detection, device):
    # Arrange
    def new_apply(self, img, bboxes):
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img), bboxes=bboxes)
            img = Image.fromarray(transformed['image'])
        return img, bboxes
    vision_data = coco_train_visiondata.get_augmented_dataset(albumentations.ShiftScaleRotate(p=1))
    dataset: CocoDataset = vision_data.data_loader.dataset
    dataset.apply_transform = types.MethodType(new_apply, dataset)
    # Act & Assert
    check = RobustnessReport()
    msg = r'Found that labels have not been affected by adding augmentation to field "transforms". This might be ' \
          r'a problem with the implementation of `Dataset.__getitem__`. label value: .*'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection,
                                             device=device),
                raises(DeepchecksValueError, msg))


def test_dataset_not_augmenting_data(coco_train_visiondata: COCOData, trained_yolov5_object_detection, device):
    # Arrange
    def new_apply(self, img, bboxes):
        return img, bboxes
    vision_data = coco_train_visiondata.get_augmented_dataset(albumentations.ShiftScaleRotate(p=1))
    dataset: CocoDataset = vision_data.data_loader.dataset
    dataset.apply_transform = types.MethodType(new_apply, dataset)

    # Act & Assert
    check = RobustnessReport()
    msg = r'Found that images have not been affected by adding augmentation to field "transforms". This might be a ' \
          r'problem with the implementation of Dataset.__getitem__'
    assert_that(calling(check.run).with_args(vision_data, trained_yolov5_object_detection,
                                             device=device),
                raises(DeepchecksValueError, msg))
