import typing as t

import torch
from torch import nn
from torch.utils.data import DataLoader
from hamcrest import (
    assert_that,
    calling,
    raises,
    equal_to,
    has_properties,
    has_property,
    instance_of,
)

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.errors import ModelValidationError
from deepchecks.core.errors import DatasetValidationError
from deepchecks.vision.base import Context
from deepchecks.vision.dataset import VisionData
from deepchecks.vision.utils import ClassificationLabelFormatter
from deepchecks.vision.utils import DetectionLabelFormatter
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist


def test_vision_data_number_of_classes_inference():
    dataset = t.cast(VisionData, mnist.load_dataset(train=True, object_type='Dataset'))
    assert_that(dataset.get_num_classes(), equal_to(10))


def test_vision_data_sample_loader():
    loader = t.cast(DataLoader, mnist.load_dataset(train=True, object_type='DataLoader'))
    dataset = VisionData(loader, num_classes=10, sample_size=100)
    samples = list(iter(dataset.sample_data_loader))

    assert_that(len(samples), equal_to(100))
    
    for s in samples:
        assert_that(len(s), equal_to(2))
        
        x, y = s
        assert_that(x, instance_of(torch.Tensor))
        assert_that(y, instance_of(torch.Tensor))
        assert_that(x.shape, equal_to((1, 1, 28, 28)))
        assert_that(y, equal_to((1,)))


def test_vision_data_task_type_inference():


