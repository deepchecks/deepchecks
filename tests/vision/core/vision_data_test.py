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
import imgaug.augmenters as iaa
import albumentations as A

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.transformations import AlbumentationsTransformations, ImgaugTransformations
from hamcrest import assert_that, instance_of, equal_to, calling, raises

from tests.vision.vision_conftest import *
from deepchecks.vision import VisionData


def test_get_transforms_type_albumentations(mnist_dataset_train):
    # Act
    transform_handler = mnist_dataset_train.get_transform_type()
    # Assert
    assert_that(transform_handler, instance_of(AlbumentationsTransformations.__class__))


def test_add_augmentation_albumentations(mnist_dataset_train):
    # Arrange
    copy_dataset = mnist_dataset_train.copy()
    augmentation = A.CenterCrop(1, 1)
    # Act
    copy_dataset.add_augmentation(augmentation)
    # Assert
    batch = next(iter(copy_dataset.get_data_loader()))
    data_sample = batch[0][0]
    assert_that(data_sample.numpy().shape, equal_to((1, 1, 1)))


def test_add_augmentation_albumentations_wrong_type(mnist_dataset_train):
    # Arrange
    copy_dataset = mnist_dataset_train.copy()
    augmentation = iaa.CenterCropToFixedSize(1, 1)
    # Act & Assert
    msg = r'Transforms is of type albumentations, can\'t add to it type CenterCropToFixedSize'
    assert_that(calling(copy_dataset.add_augmentation).with_args(augmentation),
                raises(DeepchecksValueError, msg))


def test_get_transforms_type_imgaug(mnist_dataset_train_imgaug):
    # Act
    transform_handler = mnist_dataset_train_imgaug.get_transform_type()
    # Assert
    assert_that(transform_handler, instance_of(ImgaugTransformations.__class__))


def test_add_augmentation_imgaug(mnist_dataset_train_imgaug):
    # Arrange
    copy_dataset = mnist_dataset_train_imgaug.copy()
    augmentation = iaa.CenterCropToFixedSize(1, 1)
    # Act
    copy_dataset.add_augmentation(augmentation)
    # Assert
    batch = next(iter(copy_dataset.get_data_loader()))
    data_sample = batch[0][0]
    assert_that(data_sample.numpy().shape, equal_to((1, 1, 1)))


def test_add_augmentation_imgaug_wrong_type(mnist_dataset_train_imgaug):
    # Arrange
    copy_dataset = mnist_dataset_train_imgaug.copy()
    augmentation = A.CenterCrop(1, 1)
    # Act & Assert
    msg = r'Transforms is of type imgaug, can\'t add to it type CenterCrop'
    assert_that(calling(copy_dataset.add_augmentation).with_args(augmentation),
                raises(DeepchecksValueError, msg))


def test_transforms_field_not_exists(mnist_data_loader_train):
    # Arrange
    data = VisionData(mnist_data_loader_train, transform_field='not_exists')
    # Act & Assert
    msg = r'Underlying Dataset instance does not contain "not_exists" attribute\. If your transformations field is ' \
          r'named otherwise, you cat set it by using "transform_field" parameter'
    assert_that(calling(data.get_transform_type).with_args(),
                raises(DeepchecksValueError, msg))
