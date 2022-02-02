"""Module defining base encoders."""
import abc
from typing import Callable, Optional, Union

from torch.utils.data import DataLoader


class BaseLabelEncoder(abc.ABC):
    """Base class for label encoders."""

    @abc.abstractmethod
    def __init__(self, label_encoder: Union[str, Callable]):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        pass

    @abc.abstractmethod
    def get_samples_per_class(self, data_loader: DataLoader):
        """Get the number of samples per class."""
        pass

    @abc.abstractmethod
    def validate_label(self, data_loader: DataLoader) -> Optional[str]:
        """Validate that the label format is in the required shape."""
        return 'Not implemented yet for tasks other than classification and object detection'
