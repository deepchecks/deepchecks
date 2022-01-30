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
from typing import Callable

from torch.utils.data import DataLoader
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

    def __init__(self, data_loader: DataLoader, num_classes: int = None, label_type: str = None,
                 label_transformer: Callable = None):
        self._data = data_loader

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
