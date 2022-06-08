# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The vision/dataset module containing the vision Dataset class and its functions."""
from abc import abstractmethod
from typing import List, Union

import torch

from deepchecks.core.errors import DeepchecksNotImplementedError, ValidationError
from deepchecks.vision.vision_data import TaskType, VisionData


class ClassificationData(VisionData):
    """The ClassificationData class is used to load and preprocess data for a classification task.

    It is a subclass of the VisionData class. The ClassificationData class is containing additional data and general
    methods intended for easily accessing metadata relevant for validating a computer vision classification ML models.
    """

    @property
    def task_type(self) -> TaskType:
        """Return the task type (classification)."""
        return TaskType.CLASSIFICATION

    @abstractmethod
    def batch_to_labels(self, batch) -> torch.Tensor:
        """Extract the labels from a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        torch.Tensor
            The labels extracted from the batch. The labels should be in a tensor format of shape (N,), where N is the
            number of samples in the batch. See the notes for more info.

        Examples
        --------
        >>> def batch_to_labels(self, batch):
        ...     return batch[1]

        Notes
        -----
        The accepted label format for classification is a tensor of shape (N,), when N is the number of samples.
        Each element is an integer representing the class index.
        """
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> torch.Tensor:
        """Return the predictions of the model on a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.
        model : torch.nn.Module
            The model to use for inference.
        device : torch.device
            The device to use for inference.

        Returns
        -------
        torch.Tensor
            The predictions of the model on the batch. The predictions should be in a OHE tensor format of shape
            (N, n_classes), where N is the number of samples in the batch.

        Examples
        --------
        >>> import torch.nn.functional as F
        ...
        ...
        ... def infer_on_batch(self, batch, model, device):
        ...     logits = model.to(device)(batch[0].to(device))
        ...     return F.softmax(logits, dim=1)

        Notes
        -----
        The accepted prediction format for classification is a tensor of shape (N, n_classes), where N is the number of
        samples. Each element is an array of length n_classes that represent the probability of each class.
        """
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    def get_classes(self, batch_labels: Union[List[torch.Tensor], torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        return batch_labels.reshape(-1, 1).tolist()

    def validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch
        """
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, torch.Tensor):
            raise ValidationError('Check requires classification label to be a torch.Tensor')
        label_shape = labels.shape
        if len(label_shape) != 1:
            raise ValidationError('Check requires classification label to be a 1D tensor')

    @staticmethod
    def validate_infered_batch_predictions(batch_predictions, n_classes: int = None, eps: float = 1e-3):
        """
        Validate the infered predictions from the batch.

        Parameters
        ----------
        batch_predictions : t.Any
            The infered predictions from the batch
        n_classes : int , default: None
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3

        Raises
        ------
        ValidationError
            If predictions format is invalid
        DeepchecksNotImplementedError
            If infer_on_batch not implemented
        """
        if not isinstance(batch_predictions, torch.Tensor):
            raise ValidationError('Check requires classification predictions to be a torch.Tensor')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise ValidationError('Check requires classification predictions to be a 2D tensor')
        if n_classes and pred_shape[1] != n_classes:
            raise ValidationError(f'Check requires classification predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(dim=1) - 1) > eps):
            raise ValidationError('Check requires classification predictions to be a probability distribution and'
                                  ' sum to 1 for each row')
