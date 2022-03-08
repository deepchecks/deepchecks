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
import logging
from typing import List, Optional, Dict, Union
import numpy as np

import torch
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksNotImplementedError, DeepchecksValueError, ValidationError
from deepchecks.vision.vision_data import VisionData, TaskType

logger = logging.getLogger('deepchecks')


class ClassificationData(VisionData):
    """The ClassificationData class is used to load and preprocess data for a classification task.

    It is a subclass of the VisionData class. The ClassificationData class is containing additional data and general
    methods intended for easily accessing metadata relevant for validating a computer vision classification ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. This is the data loader object that will be used to load the data.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 transform_field: Optional[str] = 'transforms'):

        super().__init__(data_loader, num_classes, label_map, transform_field)

        self._task_type = TaskType.CLASSIFICATION
        self._has_label = False
        try:
            self.validate_label(next(iter(self._data_loader)))
            self._has_label = True
        except DeepchecksNotImplementedError:
            logger.warning('batch_to_labels() was not implemented, some checks will not run')
        except ValidationError as ex:
            logger.warning('batch_to_labels() was not implemented correctly, '
                           f'the validiation has failed with the error: {str(ex)}')

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        """Extract the labels from a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
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
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
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
        Union[List[torch.Tensor], torch.Tensor]
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
        return batch_labels.tolist()

    def validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch
        """
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, (torch.Tensor, np.ndarray)):
            raise ValidationError('Check requires classification label to be a torch.Tensor or numpy array')
        label_shape = labels.shape
        if len(label_shape) != 1:
            raise ValidationError('Check requires classification label to be a 1D tensor')

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
        batch_predictions = self.infer_on_batch(batch, model, device)
        if not isinstance(batch_predictions, (torch.Tensor, np.ndarray)):
            raise ValidationError('Check requires classification predictions to be a torch.Tensor or numpy '
                                  'array')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise ValidationError('Check requires classification predictions to be a 2D tensor')
        if n_classes and pred_shape[1] != n_classes:
            raise ValidationError(f'Check requires classification predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(axis=1) - 1) > eps):
            raise ValidationError('Check requires classification} predictions to be a probability distribution and'
                                  ' sum to 1 for each row')
