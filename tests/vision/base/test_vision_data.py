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

import numpy as np
import pytest
import torch
from hamcrest import assert_that, calling, equal_to, has_length, is_not, raises
from torch.utils.data import DataLoader

from deepchecks.core.errors import ValidationError, DatasetValidationError
from deepchecks.vision.datasets.classification.mnist_torch import collate_without_model, IterableTorchMnistDataset
from deepchecks.vision.datasets.detection import coco_torch
from deepchecks.vision.datasets.segmentation import segmentation_coco
from deepchecks.vision.utils.test_utils import replace_collate_fn_dataloader
from deepchecks.vision.vision_data import TaskType, VisionData
from deepchecks.vision.vision_data.utils import validate_vision_data_compatibility
from tests.vision.conftest import run_update_loop


def _simple_batch_collate(batch):
    imgs, labels = zip(*batch)
    return {'images': list(imgs), 'labels': list(labels)}


def _batch_collate_only_images(batch):
    imgs, _ = zip(*batch)
    return {'images': list(imgs)}


def _batch_collate_only_labels(batch):
    _, labels = zip(*batch)
    return {'labels': list(labels)}


def test_vision_data_task_type_inference(mnist_visiondata_train, coco_visiondata_train,
                                         segmentation_coco_visiondata_train):
    # Assert
    assert_that(mnist_visiondata_train.task_type == TaskType.CLASSIFICATION)
    assert_that(coco_visiondata_train.task_type == TaskType.OBJECT_DETECTION)
    assert_that(segmentation_coco_visiondata_train.task_type == TaskType.SEMANTIC_SEGMENTATION)


def test_initialization_of_vision_data_with_bad_image_format():
    # Arrange
    loader_value_out_of_shape = DataLoader(dataset=[
        (torch.ones((3, 3)) * 3, 1),
        (torch.ones((3, 3)) * 2, 1),
    ], collate_fn=_simple_batch_collate)
    loader_value_dim_out_of_shape = DataLoader(dataset=[
        (torch.ones((2, 2, 2)) * 3, 1),
        (torch.ones((2, 2, 2)) * 2, 1),
    ], collate_fn=_simple_batch_collate)
    loader_value_out_of_scale = DataLoader(dataset=[
        (torch.ones((3, 3, 3)), 1),
        (torch.ones((3, 3, 3)), 1),
    ], collate_fn=_simple_batch_collate)

    # Assert
    assert_that(
        calling(VisionData).with_args(loader_value_out_of_shape, TaskType.CLASSIFICATION.value),
        raises(
            ValidationError,
            r'The image inside the iterable must be a 3D array.')
    )
    assert_that(
        calling(VisionData).with_args(loader_value_out_of_scale, TaskType.CLASSIFICATION.value),
        raises(
            ValidationError,
            r'Image data should be in uint8 format\(integers between 0 and 255\), found values in range \[1.0, 1.0\].')
    )
    assert_that(
        calling(VisionData).with_args(loader_value_dim_out_of_shape, TaskType.CLASSIFICATION.value),
        raises(
            ValidationError,
            r'The image inside the iterable must have 1 or 3 channels.')
    )


def test_initialization_of_vision_data_with_classification_dataset_that_contains_incorrect_labels():
    # Arrange
    loader_with_string_labels = DataLoader(dataset=[
        (torch.ones((3, 3, 3)) * 2, "a"),
        (torch.ones((3, 3, 3)) * 2, "b"),
    ], collate_fn=_simple_batch_collate)
    loader_with_labels_of_incorrect_shape = DataLoader(dataset=[
        (torch.ones((3, 3, 3)) * 2, torch.tensor([1, 2])),
        (torch.ones((3, 3, 3)) * 2, torch.tensor([2, 3])),
    ], collate_fn=_simple_batch_collate)

    # Assert
    assert_that(
        calling(VisionData).with_args(loader_with_string_labels, TaskType.CLASSIFICATION.value),
        raises(
            ValidationError,
            r'Classification label per image must be a number.')
    )
    assert_that(
        calling(VisionData).with_args(loader_with_labels_of_incorrect_shape,
                                      TaskType.CLASSIFICATION.value),
        raises(
            ValidationError,
            r'Classification label per image must be a number.')
    )


def test_vision_data_n_of_samples_per_class_inference_for_classification_dataset(mnist_visiondata_train):
    # Act
    run_update_loop(mnist_visiondata_train)

    # Assert
    assert_that(mnist_visiondata_train.number_of_images_cached, equal_to(200))
    assert_that(sorted(mnist_visiondata_train.get_observed_classes()), equal_to([str(x) for x in range(10)]))


def test_vision_cache_object_detection(coco_visiondata_train):
    # Act
    run_update_loop(coco_visiondata_train)
    expected_classes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0, 14.0, 16.0, 17.0, 20.0, 22.0, 23.0,
                        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 39.0, 40.0, 41.0,
                        42.0, 43.0, 44.0, 45.0, 46.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 55.0, 56.0, 57.0, 58.0, 59.0,
                        60.0, 62.0, 65.0, 67.0, 68.0, 69.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 79.0]

    # Assert
    assert_that(coco_visiondata_train.number_of_images_cached,
                equal_to(len(coco_visiondata_train.batch_loader.dataset)))
    assert_that(sorted(coco_visiondata_train.get_observed_classes()),
                equal_to(sorted(coco_visiondata_train.label_map[x] for x in expected_classes)))

    # ReAct and ReAssert
    coco_visiondata_train.init_cache()
    assert_that(coco_visiondata_train.number_of_images_cached, equal_to(0))
    assert_that(len(coco_visiondata_train.get_observed_classes()), equal_to(0))


def test_vision_data_label_comparison_with_different_datasets(coco_visiondata_train, mnist_visiondata_train):
    # Act/Assert
    assert_that(
        calling(validate_vision_data_compatibility).with_args(mnist_visiondata_train, coco_visiondata_train),
        raises(DatasetValidationError,
               'Cannot compare datasets with different task types: classification and object_detection'))


def test_vision_data_format():
    coco_dataset = coco_torch.load_dataset(object_type='DataLoader')
    assert_that(calling(VisionData).with_args(coco_dataset, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       r'Batch loader batch output must be a dictionary containing a subset of the following keys:'))


def test_detection_data_bad_batch_to_label_implementation(coco_dataloader_train):
    # Arrange
    loader_empty_batch = replace_collate_fn_dataloader(coco_dataloader_train, lambda x: {'labels': []})
    loader_classification_shape = replace_collate_fn_dataloader(coco_dataloader_train, lambda x: {'labels': [8]})
    loader_good_shape = replace_collate_fn_dataloader(coco_dataloader_train, lambda x: {'labels': [torch.Tensor([])]})
    loader_incorrect_shape = replace_collate_fn_dataloader(coco_dataloader_train,
                                                           lambda x: {'labels': [torch.Tensor([[1, 2], [1, 2]])]})

    # Assert
    assert_that(calling(VisionData).with_args(loader_empty_batch, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       "The batch labels must be a non empty iterable."))
    assert_that(calling(VisionData).with_args(loader_classification_shape, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       "label for object_detection per image must be a multi dimensional array."))
    VisionData(loader_good_shape, TaskType.OBJECT_DETECTION.value)  # should not raise error
    assert_that(calling(VisionData).with_args(loader_incorrect_shape, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       r"Object detection label per image must be a sequence of 2D arrays, where each row has 5 "
                       r"columns: \[class_id, x_min, y_min, width, height\]"))


def test_detection_data_bad_batch_to_predictions_implementation(coco_dataloader_train):
    # Arrange
    loader_empty_batch = replace_collate_fn_dataloader(coco_dataloader_train, lambda x: {'predictions': 7})
    loader_classification_shape = replace_collate_fn_dataloader(coco_dataloader_train, lambda x: {'predictions': [8]})
    loader_good_shape = replace_collate_fn_dataloader(coco_dataloader_train,
                                                      lambda x: {'predictions': [torch.Tensor([])]})
    loader_incorrect_shape = replace_collate_fn_dataloader(coco_dataloader_train,
                                                           lambda x: {'predictions': [torch.Tensor([[1, 2], [1, 2]])]})

    # Assert
    assert_that(calling(VisionData).with_args(loader_empty_batch, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       "The batch predictions must be a non empty iterable."))
    assert_that(calling(VisionData).with_args(loader_classification_shape, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       "prediction for object_detection per image must be a multi dimensional array."))
    VisionData(loader_good_shape, TaskType.OBJECT_DETECTION.value)  # should not raise error
    assert_that(calling(VisionData).with_args(loader_incorrect_shape, TaskType.OBJECT_DETECTION.value),
                raises(ValidationError,
                       r"Object detection prediction per image must be a sequence of 2D arrays, where each row has"
                       r" 6 columns: \[x_min, y_min, w, h, confidence, class_id\]"))


def test_segmentation_data_bad_batch_to_label_implementation():
    # Arrange
    coco_segmentation = segmentation_coco.load_dataset(object_type='DataLoader')
    loader_empty_batch = replace_collate_fn_dataloader(coco_segmentation, lambda x: {'labels': []})
    loader_classification_shape = replace_collate_fn_dataloader(coco_segmentation, lambda x: {'labels': [8]})
    loader_good_shape = replace_collate_fn_dataloader(coco_segmentation, lambda x: {'labels': [np.ones((3, 3))]})

    # Assert
    assert_that(calling(VisionData).with_args(loader_empty_batch, TaskType.SEMANTIC_SEGMENTATION.value),
                raises(ValidationError, "The batch labels must be a non empty iterable."))
    assert_that(calling(VisionData).with_args(loader_classification_shape, TaskType.SEMANTIC_SEGMENTATION.value),
                raises(ValidationError, "label for semantic_segmentation per image must be a multi dimensional array."))
    VisionData(loader_good_shape, TaskType.SEMANTIC_SEGMENTATION.value)  # should not raise error


def test_segmentation_data_bad_batch_to_predictions_implementation():
    # Arrange
    coco_segmentation = segmentation_coco.load_dataset(object_type='DataLoader')
    loader_empty_batch = replace_collate_fn_dataloader(coco_segmentation, lambda x: {'predictions': []})
    loader_classification_shape = replace_collate_fn_dataloader(coco_segmentation, lambda x: {'predictions': [8]})

    # Assert
    assert_that(calling(VisionData).with_args(loader_empty_batch, TaskType.SEMANTIC_SEGMENTATION.value),
                raises(ValidationError, "The batch predictions must be a non empty iterable."))
    assert_that(calling(VisionData).with_args(loader_classification_shape, TaskType.SEMANTIC_SEGMENTATION.value),
                raises(ValidationError,
                       "prediction for semantic_segmentation per image must be a multi dimensional array."))


def test_exception_image_formatter(mnist_dataloader_train):
    # Arrange
    loader_bad_images = replace_collate_fn_dataloader(mnist_dataloader_train,
                                                      lambda x: {'images': Exception('test exception')})
    loader_bad_labels = replace_collate_fn_dataloader(mnist_dataloader_train,
                                                      lambda x: {'labels': Exception('test exception')})
    loader_bad_predictions = replace_collate_fn_dataloader(mnist_dataloader_train,
                                                           lambda x: {'predictions': Exception('test exception')})

    # Act & Assert
    assert_that(
        calling(VisionData).with_args(loader_bad_images, task_type=TaskType.CLASSIFICATION.value),
        raises(Exception, 'The batch images must be an iterable, received <class \'Exception\'>.'))
    assert_that(
        calling(VisionData).with_args(loader_bad_predictions, task_type=TaskType.CLASSIFICATION.value),
        raises(Exception, 'The batch predictions must be a non empty iterable.'))
    assert_that(
        calling(VisionData).with_args(loader_bad_labels, task_type=TaskType.CLASSIFICATION.value),
        raises(Exception, 'The batch labels must be a non empty iterable.'))

def mnist_collate_labels(data):
    return {'labels': collate_without_model(data)[1]}

def test_shuffling_regular_dataloader(mnist_dataloader_train):
    # Arrange
    mnist_loader_deepchecks_format = replace_collate_fn_dataloader(mnist_dataloader_train, mnist_collate_labels)
    original_batch = next(iter(mnist_loader_deepchecks_format))
    vision_data_shuffled = VisionData(mnist_loader_deepchecks_format, TaskType.CLASSIFICATION.value,
                                      reshuffle_data=True)
    shuffled_batch = next(iter(vision_data_shuffled))
    vision_data_unshuffled = VisionData(mnist_loader_deepchecks_format, TaskType.CLASSIFICATION.value,
                                        reshuffle_data=False)
    unshuffled_batch = next(iter(vision_data_unshuffled))

    # Assert
    assert_that(original_batch['labels'], equal_to(unshuffled_batch['labels']))
    assert_that(original_batch['labels'], is_not(equal_to(shuffled_batch['labels'])))

def test_shuffling_iterator_dataloader(mnist_iterator_visiondata_train, caplog):
    # Arrange
    loader_deepchecks_format = mnist_iterator_visiondata_train.batch_loader
    original_batch = next(iter(loader_deepchecks_format))
    vision_data = VisionData(loader_deepchecks_format, TaskType.CLASSIFICATION.value, reshuffle_data=True)
    vision_data_batch = next(iter(vision_data))

    # Assert
    assert_that(original_batch['labels'], equal_to(vision_data_batch['labels'])) # no shuffling happened
    assert_that(caplog.records, has_length(1))
    assert_that(caplog.records[0].message, equal_to('Shuffling is not supported for received batch loader. '
                                                    'Make sure that your provided batch loader is indeed shuffled '
                                                    'and set shuffle_batch_loader=False'))


def test_shuffling_tf_dataset(tf_coco_visiondata_train, caplog):
    # Arrange
    loader_deepchecks_format = tf_coco_visiondata_train.batch_loader
    original_batch = next(iter(loader_deepchecks_format))
    vision_data = VisionData(loader_deepchecks_format, TaskType.OBJECT_DETECTION.value, reshuffle_data=True)
    vision_data_batch = next(iter(vision_data))

    # Assert
    assert_that(list(original_batch['labels'][0][0]), equal_to(list(vision_data_batch['labels'][0][0]))) # no shuffling happened
    assert_that(caplog.records, has_length(1))
    assert_that(caplog.records[0].message, equal_to('Shuffling for tensorflow datasets is not supported. '
                                                    'Make sure that the data used to create the Dataset was shuffled '
                                                    'beforehand and set shuffle_batch_loader=False'))
