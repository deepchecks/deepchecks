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
"""Module defining base encoders."""
import abc
from typing import Callable, Optional, Union

from torch.utils.data import DataLoader


class BaseLabelFormatter(abc.ABC):
    """Base class for label encoders."""

    @abc.abstractmethod
    def __init__(self, label_formatter: Union[str, Callable]):
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


class BasePredictionFormatter(abc.ABC):
    """Base class for prediction encoders."""

    @abc.abstractmethod
    def __init__(self, detection_formatter: Union[str, Callable]):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        pass

    @abc.abstractmethod
    def validate_prediction(self, batch_predictions, n_classes: int, eps: float = 1e-3):
        """Validate that the prediction format is in the required shape."""
        return 'Not implemented yet for tasks other than classification and object detection'
