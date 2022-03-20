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
import torch
from torch import nn
from hamcrest import (
    assert_that,
    calling,
    raises,
    equal_to,
    has_properties,
    has_property,
    instance_of,
    same_instance,
    all_of,
    is_
)

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksValueError, ValidationError, ModelValidationError
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.errors import DatasetValidationError
from deepchecks.vision.base_checks import Context
from deepchecks.vision import ClassificationData, DetectionData
from deepchecks.vision.vision_data import TaskType, VisionData


def test_vision_context_initialization_for_classification_task(mnist_dataset_train, mnist_dataset_test,
                                                               trained_mnist, device):
    # Act
    context = Context(
        train=mnist_dataset_train,
        test=mnist_dataset_test,
        model=trained_mnist,
        model_name='MNIST',
        device=device
    )

    # Assert
    assert_that(context, has_properties({
        'train': instance_of(ClassificationData),
        'test': instance_of(ClassificationData),
        'model': same_instance(trained_mnist),
        'model_name': equal_to('MNIST'),
        'device': all_of(
            instance_of(torch.device),
            has_property('type', equal_to(device.type))
        )
    }))


def test_vision_context_initialization_for_object_detection_task(coco_train_visiondata, coco_test_visiondata,
                                                                 trained_yolov5_object_detection, device):
    # Act
    context = Context(
        train=coco_train_visiondata,
        test=coco_test_visiondata,
        model=trained_yolov5_object_detection,
        model_name='COCO',
        device=device
    )

    # Assert
    assert_that(context, has_properties({
        'train': instance_of(DetectionData),
        'test': instance_of(DetectionData),
        'model': same_instance(trained_yolov5_object_detection),
        'model_name': equal_to('COCO'),
        'device': all_of(
            instance_of(torch.device),
            has_property('type', device.type)
        )
    }))


def test_vision_context_initialization_with_datasets_from_different_tasks(mnist_dataset_train, coco_train_visiondata):
    # Assert
    assert_that(
        calling(Context).with_args(train=coco_train_visiondata, test=mnist_dataset_train),
        raises(
            ValidationError,
            r'Datasets required to have same label type')
    )


def test_that_vision_context_raises_exception_for_unset_properties(mnist_dataset_train):
    # Arrange
    context = Context(train=mnist_dataset_train)

    # Act
    assert_that(
        calling(lambda: context.test),
        raises(
            DeepchecksNotSupportedError,
            r'Check is irrelevant for Datasets without test dataset')
    )
    assert_that(
        calling(lambda: context.model),
        raises(
            DeepchecksNotSupportedError,
            r'Check is irrelevant for Datasets without model')
    )


def test_empty_context_initialization():
    assert_that(
        calling(Context).with_args(model_name="Name", ),
        raises(
            DeepchecksValueError,
            r'At least one dataset \(or model\) must be passed to the method\!')
    )


def test_context_initialization_with_test_dataset_only(coco_test_visiondata):
    assert_that(
        calling(Context).with_args(model_name="Name", test=coco_test_visiondata),
        raises(
            DatasetValidationError,
            r"Can't initialize context with only test\. if you have single dataset, "
            r"initialize it as train")
    )


def test_context_initialization_with_train_dataset_only(coco_train_visiondata):
    Context(model_name="Name", train=coco_train_visiondata)


def test_context_initialization_with_model_only(trained_mnist):
    Context(model_name="Name", model=trained_mnist)


def test_context_initialization_with_broken_model(mnist_dataset_train, mnist_dataset_test):
    # Arrange
    class BrokenModel(nn.Module):
        def __call__(self, *args, **kwargs):
            raise Exception("Unvalid arguments")

    model = BrokenModel()

    # Act & Assert
    assert_that(
        calling(Context
                ).with_args(train=mnist_dataset_train,
                            test=mnist_dataset_test,
                            model=model),
        raises(
            Exception,
            r'Unvalid arguments')
    )


def test_vision_context_helper_functions(mnist_dataset_train):
    # Arrange
    context = Context(train=mnist_dataset_train)

    # Act & Assert
    assert_that(context.have_test(), is_(False))
    assert_that(context.assert_task_type(TaskType.CLASSIFICATION), is_(True))
    assert_that(calling(context.assert_task_type).with_args(TaskType.OBJECT_DETECTION),
                raises(ModelValidationError, 'Check is irrelevant for task of type TaskType.CLASSIFICATION'))

    assert_that(context.get_data_by_kind(DatasetKind.TRAIN), instance_of(ClassificationData))
    assert_that(calling(context.get_data_by_kind).with_args(DatasetKind.TEST), raises(DeepchecksNotSupportedError,
                r'Check is irrelevant for Datasets without test dataset'))