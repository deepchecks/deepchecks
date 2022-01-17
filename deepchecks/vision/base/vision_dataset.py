from torch.utils.data import DataLoader
import logging

from deepchecks.errors import DeepchecksValueError

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