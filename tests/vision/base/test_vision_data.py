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

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.dataset import VisionData
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.utils import ClassificationLabelFormatter
from deepchecks.vision.utils import DetectionLabelFormatter
from deepchecks.vision.utils.base_formatters import BaseLabelFormatter
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist


def test_vision_data_number_of_classes_inference():
    dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='VisionData'))
    assert_that(dataset.n_of_classes, equal_to(10))


def test_vision_data_task_type_inference():
    # Arrange
    mnist_loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type='DataLoader'))
    coco_loader = t.cast(DataLoader, coco.load_dataset(train=True, object_type='DataLoader'))

    class CustomLabelFormatter(BaseLabelFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(label_formatter=None)
        def __call__(self, x):
            return x
        def validate_label(self, data_loader):
            return
        def get_samples_per_class(self, *args, **kwargs):
            return {}
        def get_classes(self, batch_labels):
            return []

    # Act
    second_classification_dataset = VisionData(mnist_loader, label_formatter=ClassificationLabelFormatter(lambda x: x))
    detection_dataset = VisionData(coco_loader, label_formatter=DetectionLabelFormatter(lambda x: x))
    dataset_with_custom_formatter = VisionData(mnist_loader, label_formatter=CustomLabelFormatter())

    # Assert
    assert_that(second_classification_dataset.task_type == TaskType.CLASSIFICATION)
    assert_that(detection_dataset.task_type == TaskType.OBJECT_DETECTION)
    assert_that(dataset_with_custom_formatter.task_type is None)


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

    # Act
    first_dataset = VisionData(
        loader_with_string_labels,
        label_formatter=ClassificationLabelFormatter()
    )
    second_dataset = VisionData(
        loader_with_labels_of_incorrect_shape,
        label_formatter=ClassificationLabelFormatter()
    )

    # Assert
    assert_that(
        calling(first_dataset.assert_label),
        raises(
            DeepchecksValueError,
            r'Check requires classification label to be a torch\.Tensor or numpy array')
    )
    assert_that(
        calling(second_dataset.assert_label),
        raises(
            DeepchecksValueError,
            r'Check requires classification label to be a 1D tensor')
    )


def test_vision_data_sample_loader():
    # Arrange
    loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type='DataLoader'))
    dataset = VisionData(loader, num_classes=10, sample_size=100)

    # Act
    samples = list(iter(dataset.sample_data_loader))

    # Assert
    assert_that(len(samples), equal_to(100))

    for s in samples:
        assert_that(len(s), equal_to(2))

        x, y = s
        assert_that(x, instance_of(torch.Tensor))
        assert_that(y, instance_of(torch.Tensor))
        assert_that(x.shape, equal_to((1, 1, 28, 28)))
        assert_that(y.shape, equal_to((1,)))


def test_vision_data_n_of_samples_per_class_inference_for_classification_dataset():
    # Arrange
    loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type="DataLoader"))
    dataset = VisionData(loader, label_formatter=ClassificationLabelFormatter())

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
    dataset = VisionData(loader, label_formatter=DetectionLabelFormatter(coco.yolo_label_formatter))
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
    coco_dataset = t.cast(VisionData, coco.load_dataset(train=True, object_type='VisionData'))
    mnist_dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='VisionData'))

    # Act/Assert
    assert_that(
        calling(coco_dataset.validate_shared_label).with_args(mnist_dataset),
        raises(
            DeepchecksValueError,
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

    first_dataset = VisionData(first_loader, label_formatter=DetectionLabelFormatter())
    second_dataset = VisionData(second_loader, label_formatter=DetectionLabelFormatter())

    # Act
    # it must not raise an error
    first_dataset.validate_shared_label(second_dataset)