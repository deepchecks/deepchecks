from torch.utils.data import DataLoader
import logging

logger = logging.getLogger('deepchecks')


class VisionDataset:

    _data: DataLoader = None

    def __init__(self, data_loader: DataLoader, label_type: str = 'object_detection'):
        self._data = data_loader
        self.label_type = label_type
