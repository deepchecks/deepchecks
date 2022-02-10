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
    all_of
)

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.errors import ModelValidationError
from deepchecks.core.errors import DatasetValidationError
from deepchecks.vision.base import Context
from deepchecks.vision.dataset import VisionData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist


def test_vision_context_initialization_for_classification_task():
    # Arrange
    train_dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='Dataset'))
    test_dataset = t.cast(VisionData, mnist.load_dataset(train=False, object_type='Dataset'))
    model = mnist.load_model()

    # Act
    context = Context(
        train=train_dataset,
        test=test_dataset,
        model=model,
        model_name='MNIST',
        device='cpu',
    )

    # Assert
    assert_that(context, has_properties({
        'train': same_instance(train_dataset),
        'test': same_instance(test_dataset),
        'model': same_instance(model),
        'model_name': equal_to('MNIST'),
        'device': all_of(
            instance_of(torch.device),
            has_property('type', equal_to('cpu'))
        )
    }))


def test_vision_context_initialization_for_object_detection_task():
    # Arrange
    train_dataset = t.cast(VisionData, coco.load_dataset(train=True, object_type='Dataset'))
    test_dataset = t.cast(VisionData, coco.load_dataset(train=False, object_type='Dataset'))
    model = coco.load_model()

    # Act
    context = Context(
        train=train_dataset,
        test=test_dataset,
        model=model,
        model_name='COCO',
        device='cpu',
    )

    # Assert
    assert_that(context, has_properties({
        'train': same_instance(train_dataset),
        'test': same_instance(test_dataset),
        'model': same_instance(model),
        'model_name': equal_to('COCO'),
        'device': all_of(
            instance_of(torch.device),
            has_property('type', equal_to('cpu'))
        )
    }))


# def test_vision_context_initialization_for_segmentation_task():
#   pass


def test_vision_context_initialization_with_datasets_from_different_tasks():
    # Act
    train_dataset = t.cast(VisionData, coco.load_dataset(train=True, object_type='Dataset'))
    test_dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='Dataset'))

    # Assert
    assert_that(
        calling(Context).with_args(train=train_dataset, test=test_dataset),
        raises(
            DeepchecksValueError,
            r'Datasets required to have same label type')
    )


def test_that_vision_context_raises_exception_for_unset_properties():
    # Arrange
    train_dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='Dataset'))
    context = Context(train=train_dataset)

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
        calling(Context).with_args(model_name="Name",),
        raises(
            DeepchecksValueError,
            r'At least one dataset \(or model\) must be passed to the method\!')
    )


def test_context_initialization_with_test_dataset_only():
    test_dataset = coco.load_dataset(object_type='Dataset')
    assert_that(
        calling(Context).with_args(model_name="Name", test=test_dataset),
        raises(
            DatasetValidationError,
            r"Can't initialize context with only test\. if you have single dataset, "
            r"initialize it as train")
    )


def test_context_initialization_with_train_dataset_only():
    train_dataset = t.cast(VisionData, coco.load_dataset(train=True, object_type='Dataset'))
    Context(model_name="Name", train=train_dataset)


def test_context_initialization_with_model_only():
    model = coco.load_model()
    Context(model_name="Name", model=model)


def test_context_initialization_with_broken_model():

    # Arrange
    class BrokenModel(nn.Module):
        def __call__(self, *args, **kwargs):
            raise Exception("Unvalid arguments")

    train_dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='Dataset'))
    test_dataset = t.cast(VisionData, mnist.load_dataset(train=False, object_type='Dataset'))
    model = BrokenModel()

    # Act
    context = Context(
        train=train_dataset,
        test=test_dataset,
        model=model
    )

    # Assert
    assert_that(
        calling(lambda: context.model),
        raises(
            ModelValidationError,
            r'Got error when trying to predict with model on dataset: .*')
    )
