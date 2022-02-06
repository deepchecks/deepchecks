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
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_formatters import BaseLabelFormatter, BasePredictionFormatter

__all__ = ['ClassificationLabelFormatter', 'ClassificationPredictionFormatter']

from ...core.errors import DeepchecksValueError


class ClassificationLabelFormatter(BaseLabelFormatter):
    """
    Class for formatting the classification label to the required format.

    Parameters
    ----------
    label_formatter : Callable
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        tensor of shape (N,), When N is the number of samples. Each element is an integer
        representing the class index.

    """

    def __init__(self, label_formatter: Callable):
        super().__init__(label_formatter)
        self.label_formatter = label_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.label_formatter(*args, **kwargs)

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
        for batch in data_loader:
            counter.update(self(batch[1].tolist()))

        return counter

    def validate_label(self, data_loader: DataLoader) -> Optional[str]:
        """
        Validate the label.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to get the samples per class from.

        Returns
        -------
        Optional[str]
            None if the label is valid, otherwise a string containing the error message.

        """
        batch = next(iter(data_loader))
        if len(batch) != 2:
            return 'Check requires dataloader to return tuples of (input, label).'

        label_batch = self(batch[1])
        if not isinstance(label_batch, (torch.Tensor, np.ndarray)):
            return 'Check requires classification label to be a torch.Tensor or numpy array'
        label_shape = label_batch.shape
        if len(label_shape) != 1:
            return 'Check requires classification label to be a 1D tensor'


class ClassificationPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the classification prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch of predictions and returns the encoded predictions in the following format:
        tensor of shape (N, n_classes), When N is the number of samples. Each element is an array of length n_classes
        that represent the probability of each class.

    """

    def __init__(self, prediction_formatter: Callable):
        super().__init__(prediction_formatter)
        self.prediction_formatter = prediction_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.prediction_formatter(*args, **kwargs)

    def validate_prediction(self, batch_predictions, n_classes: int, eps: float = 1e-3):
        """
        Validate the prediction.

        Parameters
        ----------
        batch_predictions : t.Any
            Model prediction for a batch (output of model(batch[0]))
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
        if not isinstance(batch_predictions, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires classification predictions to be a torch.Tensor or numpy '
                                       'array')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise DeepchecksValueError('Check requires classification predictions to be a 2D tensor')
        if pred_shape[1] != n_classes:
            raise DeepchecksValueError(f'Check requires classification predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(axis=1) - 1) > eps):
            raise DeepchecksValueError('Check requires classification} predictions to be a probability distribution and'
                                       ' sum to 1 for each row')
