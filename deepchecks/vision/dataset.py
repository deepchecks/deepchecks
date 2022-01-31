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
        PyTorch DataLoader object
    label_type : str
        Type of label. Must be one of the following: 'classification', 'object_detection'.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_transformer : Callable, optional
        A callable, transforming a batch of labels returned by the dataloader to a batch of labels in the desired
        format.

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
                 seed: int = 0):
        self._data = data_loader

        if label_transformer is None:
            self.label_transformer = lambda x: x
        else:
            self.label_transformer = label_transformer

        valid_label_types = [tt.value for tt in TaskType]
        if label_type in valid_label_types:
            self.label_type = label_type
        else:
            raise DeepchecksValueError(f'Invalid label type: {label_type}, must be one of {valid_label_types}.')

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self._samples_per_class = None
        self._label_valid = self.label_valid()  # Will be either none if valid, or string with error
        # Sample dataset properties
        self._sample_data_loader = None
        self._sample_labels = None
        self._sample_size = sample_size
        self._seed = seed

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
                    counter.update(self.label_transformer(next(iter(self._data))[1].tolist()))
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

    @property
    def sample_data_loader(self) -> DataLoader:
        """Return sample of the data."""
        if self._sample_data_loader is None:
            self._sample_data_loader = create_sample_loader(self._data, self._sample_size, self._seed)
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
        # Getting first sample of data
        batch = next(iter(self.get_data_loader()))
        if len(batch) != 2:
            return 'Check requires dataset to have a label'

        label_batch = self.label_transformer(batch[1])
        if self.label_type == TaskType.CLASSIFICATION.value:
            if not isinstance(label_batch, (torch.Tensor, np.ndarray)):
                return f'Check requires {self.label_type} label to be a torch.Tensor or numpy array'
            label_shape = label_batch.shape
            if len(label_shape) != 1:
                return f'Check requires {self.label_type} label to be a 1D tensor'
        elif self.label_type == TaskType.OBJECT_DETECTION.value:
            if not isinstance(label_batch, list):
                return f'Check requires {self.label_type} label to be a list with an entry for each sample'
            if len(label_batch) == 0:
                return f'Check requires {self.label_type} label to be a non-empty list'
            if not isinstance(label_batch[0], (torch.Tensor, np.ndarray)):
                return f'Check requires {self.label_type} label to be a list of torch.Tensor or numpy array'
            if len(label_batch[0].shape) != 2:
                return f'Check requires {self.label_type} label to be a list of 2D tensors'
            if label_batch[0].shape[1] != 5:
                return f'Check requires {self.label_type} label to be a list of 2D tensors, when ' \
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

        if self.get_label_shape() != other.get_label_shape():
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

        return obj


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

        samples_dataset = InMemoryDataset(samples_data)
        return DataLoader(samples_dataset, generator=generator(), sampler=SequentialSampler(samples_data),
                          **common_props_to_copy)
    else:
        length = len(dataset)
        return DataLoader(dataset, generator=generator(),
                          sampler=FixedSampler(length, seed, sample_size), **common_props_to_copy)
