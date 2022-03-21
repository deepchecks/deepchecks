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
#
import itertools
import typing as t

import torch
from torch.utils.data import DataLoader
from hamcrest import (
    assert_that,
    calling,
    raises,
    equal_to,
    has_entries,
    instance_of,
    all_of,
    contains_exactly, not_
)
import albumentations as A
import imgaug.augmenters as iaa

from deepchecks.core.errors import ValidationError, DeepchecksValueError
from deepchecks.vision.classification_data import ClassificationData
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.datasets.classification.mnist import MNISTData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist
from deepchecks.vision.datasets.detection.coco import COCOData
from deepchecks.vision.detection_data import DetectionData
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.utils.transformations import AlbumentationsTransformations, ImgaugTransformations


class SimpleDetectionData(DetectionData):
    def batch_to_labels(self, batch):
        return batch[1]


class SimpleClassificationData(ClassificationData):
    def batch_to_labels(self, batch):
        return batch[1]


def test_vision_data_number_of_classes_inference():
    dataset = t.cast(MNISTData, mnist.load_dataset(train=True, object_type='VisionData'))
    assert_that(dataset.num_classes, equal_to(10))


def test_vision_data_task_type_inference():
    # Arrange
    mnist_loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type='DataLoader'))
    coco_loader = t.cast(DataLoader, coco.load_dataset(train=True, object_type='DataLoader'))

    # Act
    second_classification_dataset = SimpleClassificationData(mnist_loader)
    detection_dataset = SimpleDetectionData(coco_loader)
    base_dataset = VisionData(mnist_loader)

    # Assert
    assert_that(second_classification_dataset.task_type == TaskType.CLASSIFICATION)
    assert_that(detection_dataset.task_type == TaskType.OBJECT_DETECTION)
    assert_that(base_dataset.task_type == TaskType.OTHER)


def test_initialization_of_vision_data_with_classification_dataset_that_contains_incorrect_labels():
    # Arrange
    loader_with_string_labels = DataLoader(dataset=[
        (torch.tensor([[1,2,3],[1,2,3],[1,2,3]]), "1"),
        (torch.tensor([[1,2,3],[1,2,3],[1,2,3]]), "2"),
    ])
    loader_with_labels_of_incorrect_shape = DataLoader(dataset=[
        (torch.tensor([[1,2,3],[1,2,3],[1,2,3]]), torch.tensor([1,2])),
        (torch.tensor([[1,2,3],[1,2,3],[1,2,3]]), torch.tensor([2,3])),
    ])
    bad_type_data = SimpleClassificationData(loader_with_string_labels)
    bad_shape_data = SimpleClassificationData(loader_with_labels_of_incorrect_shape)
    # Assert
    assert_that(
        calling(bad_type_data.validate_label).with_args(next(iter(loader_with_string_labels))),
        raises(
            ValidationError,
            r'Check requires classification label to be a torch\.Tensor')
    )
    assert_that(
        calling(bad_shape_data.validate_label).with_args(next(iter(loader_with_labels_of_incorrect_shape))),
        raises(
            ValidationError,
            r'Check requires classification label to be a 1D tensor')
    )


def test_vision_data_n_of_samples_per_class_inference_for_classification_dataset():
    # Arrange
    loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type="DataLoader"))
    dataset = SimpleClassificationData(loader)

    real_n_of_samples = {}
    for index in range(len(loader.dataset)):
        X, y = loader.dataset[index]
        real_n_of_samples[y] = 1 + real_n_of_samples.get(y, 0)

    # Act
    dataset.init_cache()
    for batch in dataset:
        dataset.update_cache(dataset.batch_to_labels(batch))

    # Assert
    assert_that(
        dataset.n_of_samples_per_class,
        all_of(instance_of(dict), has_entries(real_n_of_samples))
    )


def test_vision_data_n_of_samples_per_class_inference_for_detection_dataset():
    # Arrange
    loader = t.cast(DataLoader, coco.load_dataset(train=True, object_type="DataLoader"))

    real_n_of_samples = {}
    for index in range(len(loader.dataset)):
        _, y = loader.dataset[index]
        for bbox in y:
            clazz = bbox[4].item()
            real_n_of_samples[clazz] = 1 + real_n_of_samples.get(clazz, 0)

    # Act
    dataset = coco.COCOData(loader)
    dataset.init_cache()
    for batch in dataset:
        dataset.update_cache(dataset.batch_to_labels(batch))

    # Assert
    assert_that(
        dataset.n_of_samples_per_class,
        all_of(instance_of(dict), has_entries(real_n_of_samples))
    )


# def test_vision_data_n_of_samples_per_class_inference_for_segmentation_dataset():
#     # TODO:
#     pass


def test_vision_data_label_comparison_with_different_datasets():
    # Arrange
    coco_dataset = t.cast(COCOData, coco.load_dataset(train=True, object_type='VisionData'))
    mnist_dataset = t.cast(MNISTData, mnist.load_dataset(train=True, object_type='VisionData'))

    # Act/Assert
    assert_that(
        calling(coco_dataset.validate_shared_label).with_args(mnist_dataset),
        raises(
            ValidationError,
            r'Datasets required to have same label type')
    )


def test_vision_data_label_comparison_for_detection_task():
    # Arrange
    loader = t.cast(DataLoader, coco.load_dataset(train=True, object_type="DataLoader"))

    first_X, first_label = loader.dataset[0]
    second_X, second_label = None, None

    for index in range(len(loader.dataset)):
        X, y = loader.dataset[index]
        if len(y) != len(first_label):
            second_X, second_label = X, y

    assert second_label is not None, "All images have same number of bboxes, cannot perform the test"

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    first_loader = DataLoader([(first_X, first_label),], collate_fn=batch_collate)
    second_loader = DataLoader([(second_X, second_label),], collate_fn=batch_collate)

    first_dataset = SimpleDetectionData(first_loader)
    second_dataset = SimpleDetectionData(second_loader)

    # Act
    # it must not raise an error
    first_dataset.validate_shared_label(second_dataset)


def test_get_transforms_type_albumentations(mnist_dataset_train):
    # Act
    transform_handler = mnist_dataset_train.get_transform_type()
    # Assert
    assert_that(transform_handler, instance_of(AlbumentationsTransformations.__class__))


def test_add_augmentation_albumentations(mnist_dataset_train: VisionData):
    # Arrange
    augmentation = A.CenterCrop(1, 1)
    # Act
    copy_dataset = mnist_dataset_train.get_augmented_dataset(augmentation)
    # Assert
    batch = next(iter(copy_dataset.data_loader))
    data_sample = batch[0][0]
    assert_that(data_sample.numpy().shape, equal_to((1, 1, 1)))


def test_add_augmentation_albumentations_wrong_type(mnist_dataset_train):
    # Arrange
    copy_dataset = mnist_dataset_train.copy()
    augmentation = iaa.CenterCropToFixedSize(1, 1)
    # Act & Assert
    msg = r'Transforms is of type albumentations, can\'t add to it type CenterCropToFixedSize'
    assert_that(calling(copy_dataset.get_augmented_dataset).with_args(augmentation),
                raises(DeepchecksValueError, msg))


def test_get_transforms_type_imgaug(mnist_dataset_train_imgaug):
    # Act
    transform_handler = mnist_dataset_train_imgaug.get_transform_type()
    # Assert
    assert_that(transform_handler, instance_of(ImgaugTransformations.__class__))


def test_add_augmentation_imgaug(mnist_dataset_train_imgaug: VisionData):
    # Arrange
    augmentation = iaa.CenterCropToFixedSize(1, 1)
    # Act
    copy_dataset = mnist_dataset_train_imgaug.get_augmented_dataset(augmentation)
    # Assert
    batch = next(iter(copy_dataset.data_loader))
    data_sample = batch[0][0]
    assert_that(data_sample.numpy().shape, equal_to((1, 1, 1)))


def test_add_augmentation_imgaug_wrong_type(mnist_dataset_train_imgaug: VisionData):
    # Arrange
    copy_dataset = mnist_dataset_train_imgaug.copy()
    augmentation = A.CenterCrop(1, 1)
    # Act & Assert
    msg = r'Transforms is of type imgaug, can\'t add to it type CenterCrop'
    assert_that(calling(copy_dataset.get_augmented_dataset).with_args(augmentation),
                raises(DeepchecksValueError, msg))


def test_transforms_field_not_exists(mnist_data_loader_train):
    # Arrange
    data = VisionData(mnist_data_loader_train, transform_field='not_exists')
    # Act & Assert
    msg = r'Underlying Dataset instance does not contain "not_exists" attribute\. If your transformations field is ' \
          r'named otherwise, you cat set it by using "transform_field" parameter'
    assert_that(calling(data.get_transform_type).with_args(),
                raises(DeepchecksValueError, msg))


def test_sampler(mnist_dataset_train):
    # Act
    sampled = mnist_dataset_train.copy(n_samples=10, random_state=0)
    # Assert
    classes = list(itertools.chain(*[b[1].tolist() for b in sampled]))
    assert_that(classes, contains_exactly(4, 9, 3, 3, 8, 7, 9, 4, 8, 1))

    # Act
    sampled = mnist_dataset_train.copy(n_samples=500, random_state=0)
    # Assert
    total = sum([len(b[0]) for b in sampled])
    assert_that(total, equal_to(500))


def test_data_at_batch_of_index(mnist_dataset_train):
    # Arrange
    samples_index = 100

    i = 0
    for data, labels in mnist_dataset_train.data_loader:
        if i + len(data) >= samples_index:
            single_data = data[samples_index - i]
            single_label = labels[samples_index - i]
            single_batch = mnist_dataset_train.to_batch((single_data, single_label))
            break
        else:
            i += len(data)

    # Act
    batch = mnist_dataset_train.batch_of_index(samples_index)

    # Assert
    assert torch.equal(batch[0], single_batch[0])
    assert torch.equal(batch[1], single_batch[1])


def test_get_classes_validation_not_sequence(mnist_data_loader_train):
    # Arrange
    class TestData(MNISTData):
        def get_classes(self, batch_labels):
            return 88

    # Act
    data = TestData(mnist_data_loader_train)

    # Assert
    assert_that(
        calling(data.assert_label_formatter_valid).with_args(),
        raises(DeepchecksValueError,
               r'get_classes\(\) was not implemented correctly, the validation has failed with the error: "The classes '
               r'must be a sequence\."\. '
               r'To test your formatting use the function `validate_get_classes\(batch\)`')
    )


def test_get_classes_validation_not_contain_sequence(mnist_data_loader_train):
    # Arrange
    class TestData(MNISTData):
        def get_classes(self, batch_labels):
            return [88, [1]]

    # Act
    data = TestData(mnist_data_loader_train)

    # Assert
    assert_that(
        calling(data.assert_label_formatter_valid).with_args(),
        raises(DeepchecksValueError,
               r'get_classes\(\) was not implemented correctly, the validation has failed with the error: "The '
               r'classes sequence contain also sequences of ints as values \(sequence per sample\).". To test your '
               r'formatting use the function `validate_get_classes\(batch\)`')
    )


def test_get_classes_validation_not_contain_contain_int(mnist_data_loader_train):
    # Arrange
    class TestData(MNISTData):
        def get_classes(self, batch_labels):
            return [['ss'], [1]]

    # Act
    data = TestData(mnist_data_loader_train)

    # Assert
    assert_that(
        calling(data.assert_label_formatter_valid).with_args(),
        raises(DeepchecksValueError,
               r'get_classes\(\) was not implemented correctly, the validation has failed with the error: "The '
               r'samples sequence most contain only int values.". To test your formatting use the function '
               r'`validate_get_classes\(batch\)`')
    )
