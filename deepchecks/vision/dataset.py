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
"""The vision/dataset module containing the vision Dataset class and its functions."""

from copy import copy
from enum import Enum
from collections import Counter
from typing import Callable, Optional, Union, List, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
import logging

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.transformations import TransformWrapper, get_transform_type

logger = logging.getLogger('deepchecks')

__all__ = ['TaskType', 'VisionDataset']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


class VisionDataset:
    """VisionDataset wraps a PyTorch DataLoader together with model related metadata.

    The VisionDataset class is containing additional data and methods intended for easily accessing
    metadata relevant for the training or validating of a computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. If your data loader is using IterableDataset please see note below.
    label_type : str
        Type of label. Must be one of the following: 'classification', 'object_detection'.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_transformer : Callable, optional
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

    def __init__(self,
                 data_loader: DataLoader,
                 label_type: str,
                 num_classes: Optional[int] = None,
                 label_transformer: Optional[Callable] = None,
                 sample_size: int = 1000,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transform',
                 display_transform: Optional[Callable] = None):
        self._data = data_loader

        if label_transformer is None:
            self.label_transformer = lambda x: x
        else:
            self.label_transformer = label_transformer

        valid_label_types = [tt.value for tt in TaskType]
        if label_type in valid_label_types:
            self.task_type = TaskType(label_type)
        else:
            raise DeepchecksValueError(f'Invalid label type: {label_type}, must be one of {valid_label_types}.')

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self.transform_field = transform_field
        self.display_transform = display_transform if display_transform else lambda x: x
        self._samples_per_class = None
        self._label_valid = self.label_valid()  # Will be either none if valid, or string with error
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
            if self.task_type == TaskType.CLASSIFICATION:
                counter = Counter()
                iterator = iter(self._data)
                for _ in range(len(self._data)):
                    counter.update(self.label_transformer(next(iterator)[1].tolist()))
                self._samples_per_class = counter
            elif self.task_type == TaskType.OBJECT_DETECTION:
                # Assume next(iter(self._data))[1] is a list (per sample) of numpy arrays (rows are bboxes) with the
                # first column in the array representing class
                counter = Counter()
                iterator = iter(self._data)
                for _ in range(len(self._data)):
                    list_of_arrays = self.label_transformer(next(iterator)[1])
                    class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
                    counter.update(class_list)
                self._samples_per_class = counter
            else:
                raise NotImplementedError(
                    'Not implemented yet for tasks other than classification and object detection'
                )
        return copy(self._samples_per_class)

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

    def label_valid(self) -> Union[str, bool]:
        """Validate the label of the dataset. If found problem return string describing it, else returns none."""
        batch = next(iter(self.get_data_loader()))
        if len(batch) != 2:
            return 'Check requires dataset to have a label'

        label_batch = self.label_transformer(batch[1])
        if self.task_type == TaskType.CLASSIFICATION:
            if not isinstance(label_batch, (torch.Tensor, np.ndarray)):
                return f'Check requires {self.task_type} label to be a torch.Tensor or numpy array'
            label_shape = label_batch.shape
            if len(label_shape) != 1:
                return f'Check requires {self.task_type} label to be a 1D tensor'
        elif self.task_type == TaskType.OBJECT_DETECTION:
            if not isinstance(label_batch, list):
                return f'Check requires {self.task_type} label to be a list with an entry for each sample'
            if len(label_batch) == 0:
                return f'Check requires {self.task_type} label to be a non-empty list'
            if not isinstance(label_batch[0], (torch.Tensor, np.ndarray)):
                return f'Check requires {self.task_type} label to be a list of torch.Tensor or numpy array'
            if len(label_batch[0].shape) != 2:
                return f'Check requires {self.task_type} label to be a list of 2D tensors'
            if label_batch[0].shape[1] != 5:
                return f'Check requires {self.task_type} label to be a list of 2D tensors, when ' \
                       f'each row has 5 columns: [class_id, x, y, width, height]'
        else:
            return 'Not implemented yet for tasks other than classification and object detection'

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
        return VisionDataset(new_data_loader, label_type=self.task_type.value,
                             label_transformer=self.label_transformer,
                             transform_field=self.transform_field)

    def validate_shared_properties(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Parameters
        ----------
        other : VisionDataset
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
        VisionDataset.validate_dataset(other)

        if self.is_have_label() != other.is_have_label():
            raise DeepchecksValueError('Datasets required to both either have or don\'t have labels')

        if self.task_type != other.task_type:
            raise DeepchecksValueError('Datasets required to have same label type')

        if self.get_label_shape() != other.get_label_shape():
            raise DeepchecksValueError('Datasets required to share the same label shape')

    @classmethod
    def validate_dataset(cls, obj) -> 'VisionDataset':
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
        if not isinstance(obj, VisionDataset):
            raise DeepchecksValueError('Check requires dataset to be of type VisionDataset. instead got: '
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
