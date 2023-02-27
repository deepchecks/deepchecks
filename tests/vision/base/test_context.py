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

from hamcrest import assert_that, calling, has_properties, instance_of, is_, raises

from deepchecks.core import DatasetKind
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError, ValidationError)
from deepchecks.vision.base_checks import Context
from deepchecks.vision.vision_data import TaskType, VisionData


def test_vision_context_initialization_for_classification_task(mnist_visiondata_train, mnist_visiondata_test):
    # Act
    context = Context(train=mnist_visiondata_train, test=mnist_visiondata_test)

    # Assert
    assert_that(context, has_properties({'train': instance_of(VisionData), 'test': instance_of(VisionData)}))


def test_vision_context_initialization_for_object_detection_task(coco_visiondata_train, coco_visiondata_test):
    # Act
    context = Context(train=coco_visiondata_train, test=coco_visiondata_test)

    # Assert
    assert_that(context, has_properties({'train': instance_of(VisionData), 'test': instance_of(VisionData)}))


def test_vision_context_initialization_with_datasets_from_different_tasks(mnist_visiondata_train,
                                                                          coco_visiondata_train):
    # Assert
    assert_that(
        calling(Context).with_args(train=coco_visiondata_train, test=mnist_visiondata_train),
        raises(
            DatasetValidationError,
            r'Cannot compare datasets with different task types: object_detection and classification')
    )


def test_that_vision_context_raises_exception_for_unset_properties(mnist_visiondata_train):
    # Arrange
    context = Context(train=mnist_visiondata_train)

    # Act
    assert_that(
        calling(lambda: context.test),
        raises(
            DeepchecksNotSupportedError,
            r'Check is irrelevant for Datasets without test dataset')
    )


def test_empty_context_initialization():
    assert_that(
        calling(Context).with_args(),
        raises(
            DeepchecksValueError,
            r'At least one dataset must be passed to the method\!')
    )


def test_context_initialization_with_test_dataset_only(coco_visiondata_test):
    assert_that(
        calling(Context).with_args(test=coco_visiondata_test),
        raises(
            DatasetValidationError,
            r"Can't initialize context with only test\. if you have single dataset, "
            r"initialize it as train")
    )


def test_context_initialization_with_train_dataset_only(coco_visiondata_train):
    Context(train=coco_visiondata_train)  # should pass


# Currently, no model is supported in context
# def test_context_initialization_with_training_model(mock_mnist_model):
#     trained_mnist = copy.deepcopy(mock_mnist_model.real_model)
#     trained_mnist.train()
#     assert_that(
#         calling(Context).with_args(model_name="Name", model=trained_mnist),
#         raises(
#             DatasetValidationError,
#             r'Model is not in evaluation state. Please set model training '
#             r'parameter to False or run model.eval\(\) before passing it.')
#     )
#
# def test_context_initialization_with_broken_model(mnist_visiondata_train, mnist_visiondata_test):
#     # Arrange
#     class BrokenModel(nn.Module):
#         def __call__(self, *args, **kwargs):
#             raise Exception("Invalid arguments")
#
#     model = BrokenModel()
#     model.eval()
#
#     # Act & Assert
#     assert_that(
#         calling(Context
#                 ).with_args(train=mnist_visiondata_train,
#                             test=mnist_visiondata_test,
#                             model=model),
#         raises(
#             Exception,
#             r'Invalid arguments')
#     )


def test_vision_context_helper_functions(mnist_visiondata_train):
    # Arrange
    context = Context(train=mnist_visiondata_train)

    # Act & Assert
    assert_that(context.have_test(), is_(False))
    assert_that(context.assert_task_type(TaskType.CLASSIFICATION), is_(True))
    assert_that(calling(context.assert_task_type).with_args(TaskType.OBJECT_DETECTION),
                raises(DeepchecksNotSupportedError, 'Check is irrelevant for task of type TaskType.CLASSIFICATION'))

    assert_that(context.get_data_by_kind(DatasetKind.TRAIN), instance_of(VisionData))
    assert_that(calling(context.get_data_by_kind).with_args(DatasetKind.TEST),
                raises(DeepchecksNotSupportedError, r'Check is irrelevant for Datasets without test dataset'))
