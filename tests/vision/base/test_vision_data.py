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

import albumentations as A
import imgaug.augmenters as iaa
import torch
from hamcrest import all_of, assert_that, calling, contains_exactly, equal_to, has_entries, instance_of, raises
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksNotImplementedError, DeepchecksValueError, ValidationError
from deepchecks.vision import SegmentationData
from deepchecks.vision.classification_data import ClassificationData
from deepchecks.vision.datasets.classification import mnist
from deepchecks.vision.datasets.classification.mnist import MNISTData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.detection.coco import COCOData
from deepchecks.vision.datasets.segmentation import segmentation_coco
from deepchecks.vision.detection_data import DetectionData
from deepchecks.vision.utils.transformations import AlbumentationsTransformations, ImgaugTransformations
from deepchecks.vision.vision_data import TaskType, VisionData
from tests.vision.vision_conftest import run_update_loop


class SimpleSegmentationData(SegmentationData):
    def batch_to_labels(self, batch):
        return batch[1]


class SimpleDetectionData(DetectionData):
    def batch_to_labels(self, batch):
        return [torch.cat(
            (batch[1][0][:, 4:], batch[1][0][:, :4]), 1)]


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
    segmentation_coco_loader = t.cast(DataLoader, segmentation_coco.load_dataset(train=True, object_type='DataLoader'))

    # Act
    second_classification_dataset = SimpleClassificationData(mnist_loader)
    detection_dataset = SimpleDetectionData(coco_loader)
    segmentation_dataset = SimpleSegmentationData(segmentation_coco_loader)
    base_dataset = VisionData(mnist_loader)

    # Assert
    assert_that(second_classification_dataset.task_type == TaskType.CLASSIFICATION)
    assert_that(detection_dataset.task_type == TaskType.OBJECT_DETECTION)
    assert_that(segmentation_dataset.task_type == TaskType.SEMANTIC_SEGMENTATION)
    assert_that(base_dataset.task_type == TaskType.OTHER)


def test_initialization_of_vision_data_with_classification_dataset_that_contains_incorrect_labels():
    # Arrange
    loader_with_string_labels = DataLoader(dataset=[
        (torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), "1"),
        (torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), "2"),
    ])
    loader_with_labels_of_incorrect_shape = DataLoader(dataset=[
        (torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), torch.tensor([1, 2])),
        (torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), torch.tensor([2, 3])),
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
    run_update_loop(dataset)

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
    run_update_loop(dataset)

    # Assert
    assert_that(
        dataset.n_of_samples_per_class,
        all_of(instance_of(dict), has_entries(real_n_of_samples))
    )


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

    first_loader = DataLoader([(first_X, first_label), ], collate_fn=batch_collate)
    second_loader = DataLoader([(second_X, second_label), ], collate_fn=batch_collate)

    first_dataset = SimpleDetectionData(first_loader)
    second_dataset = SimpleDetectionData(second_loader)

    # Act
    # it must not raise an error
    first_dataset.validate_shared_label(second_dataset)


def test_vision_data_label_comparison_for_segmentation_task():
    # Arrange
    loader = t.cast(DataLoader, segmentation_coco.load_dataset(train=True, object_type="DataLoader"))

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

    first_loader = DataLoader([(first_X, first_label), ], collate_fn=batch_collate)
    second_loader = DataLoader([(second_X, second_label), ], collate_fn=batch_collate)

    first_dataset = SimpleSegmentationData(first_loader)
    second_dataset = SimpleSegmentationData(second_loader)

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
    sampled = mnist_dataset_train.copy(n_samples=len(mnist_dataset_train._data_loader.dataset), random_state=0)
    # Assert
    assert_that(sampled.is_sampled(), equal_to(False))

    # Act
    sampled = mnist_dataset_train.copy(n_samples=10, random_state=0)
    # Assert
    classes = list(itertools.chain(*[b[1].tolist() for b in sampled]))
    assert_that(classes, contains_exactly(4, 9, 3, 3, 8, 7, 9, 4, 8, 1))
    assert_that(sampled.num_samples, equal_to(10))
    assert_that(sampled.is_sampled(), equal_to(True))

    # Act
    sampled = mnist_dataset_train.copy(n_samples=500, random_state=0)
    # Assert
    total = sum([len(b[0]) for b in sampled])
    assert_that(total, equal_to(500))
    assert_that(sampled.num_samples, equal_to(500))
    assert_that(sampled.is_sampled(), equal_to(True))


def test_data_at_batch_index_to_dataset_index(mnist_dataset_train):
    # Arrange
    sample_index = 100

    i = 0
    single_data = None
    single_label = None
    for data, labels in mnist_dataset_train.data_loader:
        if i + len(data) >= sample_index:
            single_data = data[sample_index - i]
            single_label = labels[sample_index - i]
            break
        else:
            i += len(data)

    # Act
    sample = mnist_dataset_train.batch_of_index(sample_index)

    # Assert
    assert torch.equal(sample[0][0], single_data)
    assert sample[1][0] == single_label


def test_get_classes_validation_not_sequence(mnist_data_loader_train):
    # Arrange
    class TestData(MNISTData):
        def get_classes(self, batch_labels):
            return 88

    # Act
    data = TestData(mnist_data_loader_train)

    # Assert
    assert_that(
        calling(data.assert_labels_valid).with_args(),
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
        calling(data.assert_labels_valid).with_args(),
        raises(DeepchecksValueError,
               r'get_classes\(\) was not implemented correctly, the validation has failed with the error: "The '
               r'classes sequence must contain as values sequences of ints \(sequence per sample\).". To test your '
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
        calling(data.assert_labels_valid).with_args(),
        raises(DeepchecksValueError,
               r'get_classes\(\) was not implemented correctly, the validation has failed with the error: "The '
               r'samples sequence must contain only int values.". To test your formatting use the function '
               r'`validate_get_classes\(batch\)`')
    )


def test_detection_data():
    coco_dataset = coco.load_dataset()
    batch = None
    model = None
    device = None
    detection_data = DetectionData(coco_dataset)
    assert_that(calling(detection_data.batch_to_labels).with_args(batch),
                raises(DeepchecksNotImplementedError, 'batch_to_labels\(\) must be implemented in a subclass'))
    assert_that(calling(detection_data.infer_on_batch).with_args(batch, model, device),
                raises(DeepchecksNotImplementedError, 'infer_on_batch\(\) must be implemented in a subclass'))


def test_detection_data_bad_implementation():
    coco_dataset = coco.load_dataset()

    class DummyDetectionData(DetectionData):
        dummy_batch = False

        def batch_to_labels(self, batch):
            if self.dummy_batch:
                return batch

            raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

        def infer_on_batch(self, batch, model, device):
            if self.dummy_batch:
                return batch

            raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    detection_data = DummyDetectionData(coco_dataset)

    detection_data.dummy_batch = True

    assert_that(calling(detection_data.validate_label).with_args(7),
                raises(ValidationError,
                       'Check requires object detection label to be a list with an entry for each sample'))
    assert_that(calling(detection_data.validate_label).with_args([]),
                raises(ValidationError,
                       'Check requires object detection label to be a non-empty list'))
    assert_that(calling(detection_data.validate_label).with_args([8]),
                raises(ValidationError,
                       'Check requires object detection label to be a list of torch.Tensor'))
    assert_that(detection_data.validate_label([torch.Tensor([])]), equal_to(None))
    assert_that(calling(detection_data.validate_label).with_args([torch.Tensor([[1, 2], [1, 2]])]),
                raises(ValidationError,
                       'Check requires object detection label to be a list of 2D tensors, when '
                       'each row has 5 columns: \[class_id, x, y, width, height\]'))

    assert_that(calling(detection_data.validate_prediction).with_args(7, None, None),
                raises(ValidationError,
                       'Check requires detection predictions to be a sequence with an entry for each sample'))
    assert_that(calling(detection_data.validate_prediction).with_args([], None, None),
                raises(ValidationError,
                       'Check requires detection predictions to be a non-empty sequence'))
    assert_that(calling(detection_data.validate_prediction).with_args([8], None, None),
                raises(ValidationError,
                       'Check requires detection predictions to be a sequence of torch.Tensor'))
    assert_that(detection_data.validate_prediction([torch.Tensor([])], None, None), equal_to(None))
    assert_that(calling(detection_data.validate_prediction).with_args([torch.Tensor([[1, 2], [1, 2]])], None, None),
                raises(ValidationError,
                       'Check requires detection predictions to be a sequence of 2D tensors, when '
                       'each row has 6 columns: \[x, y, width, height, class_probability, class_id\]'))


def test_segmentation_data():
    coco_dataset = segmentation_coco.load_dataset(object_type='DataLoader')
    batch = None
    model = None
    device = None
    segmentation_data = SegmentationData(coco_dataset)
    assert_that(calling(segmentation_data.batch_to_labels).with_args(batch),
                raises(DeepchecksNotImplementedError, 'batch_to_labels\(\) must be implemented in a subclass'))
    assert_that(calling(segmentation_data.infer_on_batch).with_args(batch, model, device),
                raises(DeepchecksNotImplementedError, 'infer_on_batch\(\) must be implemented in a subclass'))


def test_segmentation_data_bad_implementation():
    coco_dataset = segmentation_coco.load_dataset(object_type='DataLoader')

    class DummySegmentationData(SegmentationData):
        dummy_batch = False

        def batch_to_images(self, batch):
            if self.dummy_batch:
                return batch[0]

            raise DeepchecksNotImplementedError('batch_to_images() must be implemented in a subclass')

        def batch_to_labels(self, batch):
            if self.dummy_batch:
                return batch[1]

            raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

        def infer_on_batch(self, batch, model, device):
            if self.dummy_batch:
                return batch

            raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    segmentation_data = DummySegmentationData(coco_dataset)

    segmentation_data.dummy_batch = True

    # Assert label validations:
    def label_to_batch(labels, images=[torch.Tensor([1, 2, 3])]):
        return [images, labels]

    assert_that(calling(segmentation_data.validate_label).with_args(label_to_batch(7)),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation labels to be a sequence with an entry '
                       'for each sample'))
    assert_that(calling(segmentation_data.validate_label).with_args(label_to_batch([], [])),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation label to be a non-empty list'))
    assert_that(calling(segmentation_data.validate_label).with_args(label_to_batch([8])),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation label to be of type torch.Tensor'))
    assert_that(calling(segmentation_data.validate_label).with_args(label_to_batch([torch.Tensor(8)])),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation label to be of same width and height'
                       ' as the corresponding image'))

    # Assert prediction validations:
    assert_that(calling(segmentation_data.validate_prediction).with_args(7, None, None),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation predictions to be a sequence with an entry'
                       ' for each sample'))
    assert_that(calling(segmentation_data.validate_prediction).with_args([], None, None),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation predictions to be a non-empty sequence'))
    assert_that(calling(segmentation_data.validate_prediction).with_args([8], None, None),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation predictions to be of type torch.Tensor'))
    assert_that(calling(segmentation_data.validate_prediction).with_args([torch.Tensor([[1, 2], [1, 2]])], None, None),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation predictions to be a 3D tensor, but got'
                       'a tensor with 2 dimensions'))
    assert_that(
        calling(segmentation_data.validate_inferred_batch_predictions).with_args([torch.Tensor([[[1, 2], [1, 2]]])], 2),
        raises(ValidationError,
               'Deepchecks requires semantic segmentation predictions to have 2 classes'))
    assert_that(calling(segmentation_data.validate_inferred_batch_predictions).with_args(
        [torch.Tensor([[[0.1]], [[0.1]], [[0.7]]])], 3),
                raises(ValidationError,
                       'Deepchecks requires semantic segmentation predictions to be a probability '
                       'distribution and sum to 1 for each row'))

    # Assert that raises no exception:
    segmentation_data.validate_inferred_batch_predictions([torch.Tensor([[[0.1]], [[0.1]], [[0.7]]])], 3, 0.11)


def test_vision_data_initialization_from_dataset_instance(coco_train_dataloader: DataLoader):
    visiondata = VisionData.from_dataset(data=coco_train_dataloader.dataset)
    assert_that(visiondata.data_loader, instance_of(DataLoader))
    assert_that(len(coco_train_dataloader.dataset) == len(visiondata.data_loader.dataset))
    assert_that(len(list(visiondata)) == 1)
    assert_that(visiondata.num_samples == len(visiondata.data_loader.dataset))


class MyDetectionTaskData(DetectionData):
    def batch_to_images(self, batch):
        return batch[0]

    def batch_to_labels(self, batch):
        return [torch.tensor([[1, 2, 3, 4, -5]])]

    def infer_on_batch(self, batch, model, device):
        return [torch.tensor([[1, 2, 3, 4, 5, 6]])]


def test_validation_bad_batch_to_label(coco_train_dataloader: DataLoader):
    vision_data = MyDetectionTaskData(coco_train_dataloader)
    assert_that(vision_data._label_formatter_error, equal_to('batch_to_labels() was not implemented correctly, '
                                                             'the validation has failed with the error: \"Found one '
                                                             'of coordinates to be negative, check requires object '
                                                             'detection bounding box coordinates to be of format ['
                                                             'class_id, x, y, width, height].\". To test your label '
                                                             'formatting use the function `validate_label(batch)`'))
