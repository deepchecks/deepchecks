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
from typing import Callable

import numpy as np
import torch

from .base_formatters import BaseLabelFormatter, BasePredictionFormatter
from deepchecks.core.errors import DeepchecksValueError


__all__ = ['ClassificationLabelFormatter', 'ClassificationPredictionFormatter']


class ClassificationLabelFormatter(BaseLabelFormatter):
    """
    Class for formatting the classification label to the required format.

    Parameters
    ----------
    label_formatter : Callable
        Function that takes in a batch from DataLoader and returns the encoded labels from it in the following format:
        tensor of shape (N,), where N is the number of samples. Each element is an integer
        representing the class index.

    Examples
    --------
    For a given dataloader that returns the following label structure: [class_id, image_sha1].
    To transform the labels to the accepted format, we will implement the following function:

    >>> from deepchecks.vision.utils import ClassificationLabelFormatter
    ...
    ...
    ... def to_accepted_format(batch):
    ...     return batch[1][:, 0]
    ...
    ...
    ... label_formatter = ClassificationLabelFormatter(to_accepted_format)

    See Also
    --------
    ClassificationPredictionFormatter
    """

    def __init__(self, label_formatter: Callable = lambda x: x[1]):
        super().__init__(label_formatter)
        self.label_formatter = label_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.label_formatter(*args, **kwargs)

    def get_classes(self, batch_labels):
        """Get a labels batch and return classes inside it."""
        return batch_labels.tolist()

    def validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch
        """
        labels = self(batch)
        if not isinstance(labels, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires classification label to be a torch.Tensor or numpy array')
        label_shape = labels.shape
        if len(label_shape) != 1:
            raise DeepchecksValueError('Check requires classification label to be a 1D tensor')


class ClassificationPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the classification prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch from DataLoader and model, and returns the encoded predictions in the following
        format:
        tensor of shape (N, n_classes), When N is the number of samples. Each element is an array of length n_classes
        that represent the probability of each class.

    Examples
    --------
    For a given model that returns the logits of the model, without applying softmax.
    To transform the predictions to the accepted format, we will use the following function:

    >>> import torch.nn.functional as F
    ... from deepchecks.vision.utils import ClassificationPredictionFormatter
    ...
    ...
    ... def to_accepted_format(batch, model, device):
    ...     predictions = model(batch[0])
    ...     return F.softmax(predictions, dim=1)
    ...
    ...
    ... pred_formatter = ClassificationPredictionFormatter(to_accepted_format)


    See Also
    --------
    ClassificationLabelFormatter
    """

    def __init__(self, prediction_formatter: Callable = None):
        super().__init__(prediction_formatter)
        if prediction_formatter is None:
            self.prediction_formatter = lambda batch, model, device: model.to(device)(batch[0].to(device))
        else:
            self.prediction_formatter = prediction_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.prediction_formatter(*args, **kwargs)

    def validate_prediction(self, batch, model, device, n_classes: int = None, eps: float = 1e-3):
        """
        Validate the prediction.

        Parameters
        ----------
        batch : t.Any
            Batch as outputed from DataLoader
        model: t.Any
            Model to run on batch
        device: str
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
        batch_predictions = self(batch, model, device)
        if not isinstance(batch_predictions, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires classification predictions to be a torch.Tensor or numpy '
                                       'array')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise DeepchecksValueError('Check requires classification predictions to be a 2D tensor')
        if n_classes and pred_shape[1] != n_classes:
            raise DeepchecksValueError(f'Check requires classification predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(axis=1) - 1) > eps):
            raise DeepchecksValueError('Check requires classification} predictions to be a probability distribution and'
                                       ' sum to 1 for each row')
