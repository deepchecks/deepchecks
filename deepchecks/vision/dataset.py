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
from typing import Callable, List, Iterator

from torch.utils.data import DataLoader, SequentialSampler, Dataset, Sampler
import logging
import numpy as np
import torch
from torch.utils.data.sampler import T_co

from deepchecks.core.errors import DeepchecksValueError

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
    metadata relevant for the training or validating of an computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_type : str, optional
        Type of label. If not provided, will be inferred from the dataset.
    label_transformer : Callable, optional

    """

    _data: DataLoader = None
    _sample_data: DataLoader = None

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: int = None,
                 label_type: str = None,
                 label_transformer: Callable = None,
                 sample_size: int = 1000,
                 seed: int = 0):
        self._data, self._sample_data = self._create_data_loaders(data_loader, sample_size, seed)

        if label_transformer is None:
            self.label_transformer = lambda x: x
        else:
            self.label_transformer = label_transformer

        if label_type is not None:
            self.label_type = label_type
        else:
            self.label_type = self.infer_label_type()

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self._samples_per_class = None

    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            samples_per_class = self.get_samples_per_class()
            num_classes = len(samples_per_class.keys())
            self._num_classes = num_classes
        return self._num_classes

    def get_samples_per_class(self):
        """Return a dictionary containing the number of samples per class."""
        if self._samples_per_class is None:
            if self.label_type == TaskType.CLASSIFICATION.value:
                counter = Counter()
                for _ in range(len(self._data)):
                    counter.update(self.label_transformer(next(iter(self._data))[1]))
                self._samples_per_class = counter
            elif self.label_type == TaskType.OBJECT_DETECTION.value:
                # Assume next(iter(self._data))[1] is a list (per sample) of numpy arrays (rows are bboxes) with the
                # first column in the array representing class
                counter = Counter()
                for _ in range(len(self._data)):
                    list_of_arrays = self.label_transformer(next(iter(self._data))[1])
                    class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
                    counter.update(class_list)
                self._samples_per_class = counter
            else:
                raise NotImplementedError(
                    'Not implemented yet for tasks other than classification and object detection'
                )
        return copy(self._samples_per_class)

    def infer_label_type(self):
        """Infer the type of label from the dataset."""
        label_shape = self.get_label_shape()

        # Means the tensor is an array of scalars
        if len(label_shape) == 0:
            return TaskType.CLASSIFICATION.value
        else:
            return TaskType.OBJECT_DETECTION.value

    def validate_label(self):
        """Validate the label type of the dataset."""
        # Getting first sample of data
        sample = self._data.dataset[0]
        if len(sample) != 2:
            raise DeepchecksValueError('Check requires dataset to have a label')

    def get_label_shape(self):
        """Return the shape of the label."""
        self.validate_label()

        # Assuming the dataset contains a tuple of (features, label)
        return self.label_transformer(next(iter(self._data))[1])[0].shape  # first argument is batch_size

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data)

    def get_data_loader(self):
        """Return the data loader."""
        return self._data

    def validate_shared_label(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Parameters
        ----------
        other : Dataset
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

        label_shape = self.get_label_shape()
        other_label_shape = other.get_label_shape()

        if other_label_shape != label_shape:
            raise DeepchecksValueError('Check requires datasets to share the same label shape')

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
        if len(obj.get_data_loader().dataset) == 0:
            raise DeepchecksValueError('Check requires a non-empty dataset')

        return obj

    @classmethod
    def _create_data_loaders(cls, data_loader: DataLoader, sample_size: int, seed: int):
        """Create a data loader which is shuffled and data loader with only a subset of the data."""
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
        # IterableDataset doesn't work with samplers, so instead we manually copy all samples to memory and create
        # new dataset that will contain them.
        if isinstance(dataset, torch.utils.data.IterableDataset):
            iter_length = 0
            for _ in dataset:
                iter_length += 1
            np.random.seed(seed)
            sample_indices = set(np.random.choice(iter_length, size=(sample_size,), replace=False))

            samples_data = []
            for i, sample in enumerate(dataset):
                if i in sample_indices:
                    samples_data.append(sample)

            full_loader = data_loader
            samples_dataset = InMemoryDataset(samples_data)
            sample_loader = DataLoader(samples_dataset, generator=generator(), sampler=SequentialSampler(samples_data),
                                       **common_props_to_copy)
        else:
            length = len(dataset)
            full_loader = DataLoader(dataset, generator=generator(),
                                     sampler=FixedSampler(length, seed), **common_props_to_copy)
            sample_loader = DataLoader(dataset, generator=generator(),
                                       sampler=FixedSampler(length, seed, sample_size), **common_props_to_copy)

        return full_loader, sample_loader


class InMemoryDataset(Dataset):
    """Dataset implementation that gets all the data as in-memory list."""

    def __init__(self, data: List):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


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
