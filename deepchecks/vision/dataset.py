# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The vision/dataset module containing the vision Dataset class and its functions."""
from collections import Counter
from copy import copy
from enum import Enum
from typing import Optional, Union, List, Iterator, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import logging

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.transformations import TransformWrapper, get_transform_type
from deepchecks.vision.utils import ClassificationLabelFormatter, DetectionLabelFormatter
from deepchecks.vision.utils.base_formatters import BaseLabelFormatter
from deepchecks.vision.utils.data_formatters import DataFormatter

logger = logging.getLogger('deepchecks')

__all__ = ['TaskType', 'VisionData']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


class VisionData:
    """VisionData wraps a PyTorch DataLoader together with model related metadata.

    The VisionData class is containing additional data and methods intended for easily accessing
    metadata relevant for the training or validating of a computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. If your data loader is using IterableDataset please see note below.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_transformer : Union[ClassificationLabelFormatter, DetectionLabelFormatter]
        A callable, transforming a batch of labels returned by the dataloader to a batch of labels in the desired
        format.
    sample_size : int, default: 1,000
        Sample size to run the checks on.
    random_seed : int, default: 0
        Random seed used to generate the sample.

    Notes
    -----
    Accepted label formats are:
        * Classification: tensor of shape (N,), When N is the number of samples. Each element is an integer
          representing the class index.
        * Object Detection: List of length N containing tensors of shape (B, 5), where N is the number of samples,
          B is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
          x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
          the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.

    The labels returned by the data loader (e.g. by using next(iter(data_loader))[1]) should be in the specified format,
    or else the callable label_transformer should be able to transform the labels to the desired format.
    """

    _data: DataLoader = None
    label_transformer: BaseLabelFormatter = None

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_transformer: Union[ClassificationLabelFormatter, DetectionLabelFormatter] = None,
                 data_transformer: DataFormatter = None,
                 sample_size: int = 1000,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transform'):
        self._data = data_loader
        self.label_transformer = label_transformer or ClassificationLabelFormatter(lambda x: x)

        if isinstance(self.label_transformer, ClassificationLabelFormatter):
            self.task_type = TaskType.CLASSIFICATION
        elif isinstance(self.label_transformer, DetectionLabelFormatter):
            self.task_type = TaskType.OBJECT_DETECTION
        else:
            logger.warning('Unknown label transformer type was provided. Only integrity and data checks will run.'
                           'The supported label transformer types are: '
                           '[ClassificationLabelFormatter, DetectionLabelFormatter]')

        self.data_transformer = data_transformer or DataFormatter(lambda x: x)
        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self.transform_field = transform_field
        self._samples_per_class = None
        self._label_valid = self.label_transformer.validate_label(self._data)  # will contain error message if not valid
        # Sample dataset properties
        self._sample_data_loader = None
        self._sample_labels = None
        self._sample_size = sample_size
        self._random_seed = random_seed

    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            samples_per_class = self.get_samples_per_class()
            num_classes = len(samples_per_class.keys())
            self._num_classes = num_classes
        return self._num_classes

    def get_samples_per_class(self) -> Counter:
        """Return a dictionary containing the number of samples per class."""
        if self._samples_per_class is None:
            if self.task_type in [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION]:
                self._samples_per_class = self.label_transformer.get_samples_per_class(self._data)
            else:
                raise NotImplementedError(
                    'Not implemented yet for tasks other than classification and object detection'
                )
        return copy(self._samples_per_class)

    def to_display_data(self, batch):
        """Convert a batch of data outputted by the data loader to a format that can be displayed."""
        self.data_transformer.validate_data(batch)
        return self.data_transformer(batch)

    @property
    def sample_data_loader(self) -> DataLoader:
        """Return sample of the data."""
        if self._sample_data_loader is None:
            self._sample_data_loader = create_sample_loader(self._data, self._sample_size, self._random_seed)
        return self._sample_data_loader

    @property
    def sample_labels(self) -> List:
        """Return the labels of the sample data."""
        if self._sample_labels is None:
            self._sample_labels = []
            for _, label in self.sample_data_loader:
                self._sample_labels.append(label)
        return self._sample_labels

    def get_label_shape(self):
        """Return the shape of the label."""
        self.assert_label()

        # Assuming the dataset contains a tuple of (features, label)
        return self.label_transformer(next(iter(self._data))[1])[0].shape  # first argument is batch_size

    def assert_label(self):
        """Raise error if label is not exists or not valid."""
        if isinstance(self._label_valid, str):
            raise DeepchecksValueError(self._label_valid)

    def is_have_label(self) -> bool:
        """Return whether the data contains labels."""
        batch = next(iter(self.get_data_loader()))
        return len(batch) == 2

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data)

    def get_data_loader(self):
        """Return the data loader."""
        return self._data

    def get_transform_type(self) -> str:
        dataset_ref = self.get_data_loader().dataset
        try:
            transform = dataset_ref.__getattribute__(self.transform_field)
            return get_transform_type(transform)
        # If no field exists raise error
        except AttributeError:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self.transform_field} attribute")

    def wrap_transform_field(self) -> TransformWrapper:
        """Validate transform field in the dataset, and return it wrapped in TransformWrapper"""
        dataset_ref = self.get_data_loader().dataset
        try:
            transform = dataset_ref.__getattribute__(self.transform_field)
            wrapper = TransformWrapper(transform)
            dataset_ref.__setattr__(self.transform_field, wrapper)
            return wrapper
        # If no field exists raise error
        except AttributeError:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self.transform_field} attribute")

    def copy(self) -> 'VisionDataset':
        props = get_data_loader_props_to_copy(self.get_data_loader())
        props['dataset'] = copy(self.get_data_loader().dataset)
        new_data_loader = self.get_data_loader().__class__(**props)
        return VisionData(new_data_loader,
                          data_transformer=self.data_transformer,
                          label_transformer=self.label_transformer,
                          transform_field=self.transform_field)

    def validate_shared_properties(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Parameters
        ----------
        other : VisionData
            Expected to be Dataset type. dataset to compare
        Returns
        -------
        Hashable
            name of the label column
        Raises
        ------
        DeepchecksValueError
            if datasets don't have the same label
        """
        VisionData.validate_dataset(other)

        if self.is_have_label() != other.is_have_label():
            raise DeepchecksValueError('Datasets required to both either have or don\'t have labels')

        if self.task_type != other.task_type:
            raise DeepchecksValueError('Datasets required to have same label type')

        if self.get_label_shape() != other.get_label_shape():
            raise DeepchecksValueError('Datasets required to share the same label shape')

    @classmethod
    def validate_dataset(cls, obj) -> 'VisionData':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Parameters
        ----------
        obj : any
            object to validate as dataset
        Returns
        -------
        Dataset
            object that is deepchecks dataset
        """
        if not isinstance(obj, VisionData):
            raise DeepchecksValueError('Check requires dataset to be of type VisionData. instead got: '
                                       f'{type(obj).__name__}')

        return obj


class FixedSampler(Sampler):
    """Sampler which returns indices in a shuffled constant order."""

    _length: int
    _seed: int
    _indices = None

    def __init__(self, length: int, seed: int = 0, sample_size: int = None) -> None:
        super().__init__(None)
        self._length = length
        self._seed = seed
        if sample_size:
            sample_size = min(sample_size, length)
            np.random.seed(self._seed)
            self._indices = np.random.choice(self._length, size=(sample_size,), replace=False)

    def __iter__(self) -> Iterator[int]:
        if self._indices:
            for i in self._indices:
                yield i
        else:
            for i in torch.randperm(self._length, generator=torch.Generator.manual_seed(self._seed)):
                yield i

    def __len__(self) -> int:
        return len(self._indices) if self._indices else self._length


def create_sample_loader(data_loader: DataLoader, sample_size: int, seed: int):
    """Create a data loader with only a subset of the data."""
    common_props_to_copy = {
        'num_workers': data_loader.num_workers,
        'collate_fn': data_loader.collate_fn,
        'pin_memory': data_loader.pin_memory,
        'timeout': data_loader.timeout,
        'worker_init_fn': data_loader.worker_init_fn,
        'prefetch_factor': data_loader.prefetch_factor,
        'persistent_workers': data_loader.persistent_workers
    }

    generator = lambda: torch.Generator().manual_seed(seed)

    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.IterableDataset):
        raise DeepchecksValueError('Unable to create sample for IterableDataset')
    else:
        length = len(dataset)
        return DataLoader(dataset, generator=generator(),
                          sampler=FixedSampler(length, seed, sample_size), **common_props_to_copy)


def get_data_loader_props_to_copy(data_loader):
    props = {
        'num_workers': data_loader.num_workers,
        'collate_fn': data_loader.collate_fn,
        'pin_memory': data_loader.pin_memory,
        'timeout': data_loader.timeout,
        'worker_init_fn': data_loader.worker_init_fn,
        'prefetch_factor': data_loader.prefetch_factor,
        'persistent_workers': data_loader.persistent_workers
    }
    if data_loader.batch_sampler is not None:
        props['batch_sampler'] = data_loader.batch_sampler
    else:
        props['sampler'] = data_loader.sampler
    return props
