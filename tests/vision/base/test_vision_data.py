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
)

from deepchecks.core.errors import ValidationError
from deepchecks.vision.classification_data import ClassificationData
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.datasets.classification.mnist import MNISTData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist
from deepchecks.vision.datasets.detection.coco import COCOData
from deepchecks.vision.detection_data import DetectionData
from deepchecks.vision.vision_data import VisionData


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
    assert_that(base_dataset.task_type is None)


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
            r'Check requires classification label to be a torch\.Tensor or numpy array')
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
    infered_n_of_samples = dataset.n_of_samples_per_class

    # Assert
    assert_that(
        infered_n_of_samples,
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
    infered_n_of_samples = dataset.n_of_samples_per_class

    # Assert
    assert_that(
        infered_n_of_samples,
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
