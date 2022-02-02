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
"""Module for defining detection encoders."""
from collections import Counter
from typing import Callable

from torch.utils.data import DataLoader

from .base_encoders import BaseLabelEncoder
__all__ = ["ClassificationLabelEncoder", "ClassificationPredictionEncoder"]


class ClassificationLabelEncoder(BaseLabelEncoder):
    """
    Class for encoding the classification label to the required format.

    Parameters
    ----------
    label_encoder : Callable
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        tensor of shape (N,), When N is the number of samples. Each element is an integer
        representing the class index.

    """

    def __init__(self, label_encoder: Callable):
        self.label_encoder = label_encoder

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.label_encoder(*args, **kwargs)

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
            Dictionary of the number of samples per class.

        """

        counter = Counter()
        for _ in range(len(data_loader)):
            counter.update(self(next(iter(data_loader))[1].tolist()))

        return counter


class ClassificationPredictionEncoder:
    """
    Class for encoding the classification prediction to the required format.

    Parameters
    ----------
    prediction_encoder : Callable
        Function that takes in a batch of predictions and returns the encoded predictions in the following format:
        tensor of shape (N, n_classes), When N is the number of samples. Each element is an array of length n_classes
        that represent the probability of each class.

    """

    def __init__(self, prediction_encoder: Callable):
        self.prediction_encoder = prediction_encoder

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.prediction_encoder(*args, **kwargs)
