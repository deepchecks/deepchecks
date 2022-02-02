from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch.utils.data import DataLoader


class BaseLabelEncoder(ABC):
    """
    Base class for label encoders.
    """

    @abstractmethod
    def __init__(self, label_encoder: Callable):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        pass

    @abstractmethod
    def get_samples_per_class(self, data_loader: DataLoader):
        """
        Get the number of samples per class.
        """
        pass

    @abstractmethod
    def validate_label(self, data_loader: DataLoader) -> Optional[str]:
        """
        Validate that the label format is in the required shape.
        """
        return 'Not implemented yet for tasks other than classification and object detection'
