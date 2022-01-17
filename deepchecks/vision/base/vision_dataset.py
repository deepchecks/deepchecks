from torch.utils.data import DataLoader
import logging

from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.typing import Hashable

logger = logging.getLogger('deepchecks')


class VisionDataset:

    _data: DataLoader = None

    def __init__(self, data_loader: DataLoader, label_type: str = 'object_detection'):
        self._data = data_loader
        self.label_type = label_type

    def validate_label(self):
        # Getting first sample of data
        sample = self._data.dataset[0]
        if len(sample) != 2:
            raise DeepchecksValueError('Check requires dataset to have a label')

    def get_label_shape(self):
        self.validate_label()

        # Assuming the dataset contains a tuple of (features, label)
        return next(iter(self._data))[1].shape

    def __iter__(self):
        return iter(self._data)

    def get_data_loader(self):
        return self._data

    def validate_shared_label(self, other) -> Hashable:
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

        label_shape = self.get_label_shape()[0].shape
        other_label_shape = other.get_label_shape()[0].shape

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