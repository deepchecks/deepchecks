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
from collections import Counter
from typing import Callable, Union

from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksValueError


class BaseLabelFormatter(abc.ABC):
    """Base class for label encoders."""

    @abc.abstractmethod
    def __init__(self, label_formatter: Union[str, Callable]):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        pass

    def get_samples_per_class(self, data_loader: DataLoader):
        """
        Get the number of samples per class.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to get the samples per class from.

        Returns
        -------
        Counter
            Counter of the number of samples per class.
        """
        counter = Counter()
        for batch in data_loader:
            labels = self(batch)
            counter.update(self.get_classes(labels))
        return counter

    @abc.abstractmethod
    def get_classes(self, batch_labels):
        """Get a labels batch and return classes inside it."""
        pass

    @abc.abstractmethod
    def validate_label(self, batch):
        """Validate that the label is in the required format."""
        raise DeepchecksValueError('Not implemented yet for tasks other than classification and object detection')


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
    def validate_prediction(self, batch, model, device, n_classes: int = None, eps: float = 1e-3):
        """Validate that the predictions are in the required format."""
        return 'Not implemented yet for tasks other than classification and object detection'
