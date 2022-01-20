from copy import copy
from enum import Enum
from collections import Counter

from torch.utils.data import DataLoader
from torch import cat
import logging

from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.typing import Hashable

logger = logging.getLogger('deepchecks')

__all__ = ['TaskType', 'VisionDataset']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


class VisionDataset:

    _data: DataLoader = None

    def __init__(self, data_loader: DataLoader, num_classes: int = None, label_type: str = None):
        self._data = data_loader
        if label_type is not None:
            self.label_type = label_type
        else:
            self.label_type = self.infer_label_type()

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self._samples_per_class = None

    def get_num_classes(self):
        if self._num_classes is None:
            samples_per_class = self.get_samples_per_class()
            num_classes = len(samples_per_class.keys())
            self._num_classes = num_classes
        return self._num_classes

    def get_samples_per_class(self):
        if self._samples_per_class is None:
            if len(self.get_label_shape()) == 0:
                counter = Counter()
                for i in range(len(self._data)):
                    counter.update(next(iter(self._data))[1].tolist())
                self._samples_per_class = counter
            else:
                raise NotImplementedError('Not implemented yet') #TODO
        return copy(self._samples_per_class)

    def extract_label(self):
        y = []
        for i in range(len(self._data)):
            y.append(next(iter(self._data))[1])
        return cat(y, 0)

    def infer_label_type(self):
        return 'classification'

    def validate_label(self):
        # Getting first sample of data
        sample = self._data.dataset[0]
        if len(sample) != 2:
            raise DeepchecksValueError('Check requires dataset to have a label')

    def get_label_shape(self):
        self.validate_label()

        # Assuming the dataset contains a tuple of (features, label)
        return next(iter(self._data))[1].shape[1:]  # first argument is batch_size

    def __iter__(self):
        return iter(self._data)

    def get_data_loader(self):
        return self._data

    def validate_shared_label(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Args:
            other (Dataset): Expected to be Dataset type. dataset to compare

        Returns:
            Hashable: name of the label column

        Raises:
            DeepchecksValueError if datasets don't have the same label
        """
        VisionDataset.validate_dataset(other)

        label_shape = self.get_label_shape()
        other_label_shape = other.get_label_shape()

        if other_label_shape != label_shape:
            raise DeepchecksValueError('Check requires datasets to share the same label shape')

    @classmethod
    def validate_dataset(cls, obj) -> 'VisionDataset':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Args:
            obj: object to validate as dataset

        Returns:
            (Dataset): object that is deepchecks dataset
        """
        if not isinstance(obj, VisionDataset):
            raise DeepchecksValueError('Check requires dataset to be of type VisionDataset. instead got: '
                                       f'{type(obj).__name__}')
        if len(obj._data.dataset) == 0:
            raise DeepchecksValueError('Check requires a non-empty dataset')

        return obj