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
"""Module containing the SegmentationData class and its functions."""
from abc import abstractmethod
from typing import List, Sequence

import torch

from deepchecks.core.errors import DeepchecksNotImplementedError, ValidationError
from deepchecks.vision.vision_data import TaskType, VisionData


class SegmentationData(VisionData):
    """The SegmentationData class is used to load and preprocess data for a semantic segmentation task.

    It is a subclass of the VisionData class. The SegmentationData class contains additional data and general
    methods intended for easily accessing metadata relevant for validating a computer vision semantic segmentation ML
    models.
    """

    @property
    def task_type(self) -> TaskType:
        """Return the task type (semantic_segmentation)."""
        return TaskType.SEMANTIC_SEGMENTATION

    @abstractmethod
    def batch_to_labels(self, batch) -> List[torch.Tensor]:
        """Extract the labels from a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        List[torch.Tensor]
            The labels the images in the batch. The images should be in a list of length N containing
            tensors of shape (H, W), where N is the number of images, and H and W are the height and width of the
            corresponding image, and its values are the true class_ids of the corresponding pixels in that image.
            Note that the tensor should 2D, as the number of channels on the original image are irrelevant to the class.

        Examples
        --------
        >>> import numpy as np
        ...
        ...
        ... def batch_to_labels(self, batch):
        ...     # In this example, each image's label is a tensor of boolean masks, one per class_id, indicating whether
        ...     # that pixel is of that class.
        ...     # We would like to convert to a format where the function returns a single mask indicating the exact
        ...     # of each pixel:
        ...     images = batch[0]
        ...     labels = batch[1]
        ...     return_labels = []
        ...
        ...     for label, image in zip(images, labels):
        ...         # Here, class_id "0" is "background" or "no class detected"
        ...         ret_label = np.zeros((image.shape[0], image.shape[1]))
        ...         # Mask to mark which pixels are already identified as classes, in case of overlap in boolean masks
        ...         ret_label_taken_positions = np.zeros(ret_label.shape)
        ...
        ...         # Go over all masks of this image and transform them to a single one:
        ...         for i in range(len(label)):
        ...             mask = np.logical_and(np.logical_not(ret_label_taken_positions), np.array(label[i]))
        ...             ret_label += i * mask
        ...
        ...             # Update the taken positions:
        ...             ret_label_taken_positions = np.logical_or(ret_label_taken_positions, mask)
        ...         return_labels.append(ret_label)
        ...
        ...     return return_labels
        """
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Sequence[torch.Tensor]:
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
        Sequence[torch.Tensor]
            The predictions of the model on the batch. The predictions should be in a sequence of length N containing
            tensors of shape (C, H, W), where N is the number of images, H and W are the height and width of the
            corresponding image, and C is the number of classes that can be detected, each channel corresponds to a
            class_id.
            Note that the values of dimension C are the probabilities for each class and should sum to 1.

        Examples
        --------
        >>> import torch
        ...
        ...
        ... def infer_on_batch(self, batch, model, device):
        ...     # Converts prediction received as (H, W, C) format to (C, H, W) format:
        ...     return_list = []
        ...
        ...     predictions = model(batch[0])
        ...     for single_image_tensor in predictions:
        ...         single_image_tensor = torch.transpose(single_image_tensor, 0, 2)
        ...         single_image_tensor = torch.transpose(single_image_tensor, 1, 2)
        ...         return_list.append(single_image_tensor)
        ...
        ...     return return_list
        """
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    def get_classes(self, batch_labels: List[torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        return [torch.unique(tensor).type(torch.IntTensor).tolist() for tensor in batch_labels]

    def validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch

        Raises
        ------
        DeepchecksValueError
            If labels format is invalid
        DeepchecksNotImplementedError
            If batch_to_labels not implemented
        """
        images = self.batch_to_images(batch)
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, Sequence) or not len(labels) == len(images):
            raise ValidationError('Deepchecks requires semantic segmentation labels to be a sequence with an entry for '
                                  'each sample')
        if len(labels) == 0:
            raise ValidationError('Deepchecks requires semantic segmentation label to be a non-empty list')
        for image, label in zip(images, labels):
            if not isinstance(label, torch.Tensor):
                raise ValidationError('Deepchecks requires semantic segmentation label to be of type torch.Tensor')
            if not label.shape == image.shape[:2]:
                raise ValidationError('Deepchecks requires semantic segmentation label to be of same width and height'
                                      ' as the corresponding image')

    @staticmethod
    def validate_inferred_batch_predictions(batch_predictions, n_classes: int = None, eps: float = 1e-3):
        """
        Validate the inferred predictions from the batch.

        Parameters
        ----------
        batch_predictions : t.Any
            The inferred predictions from the batch
        n_classes : int , default: None
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3


        Raises
        ------
        ValidationError
            If predictions format is invalid
        """
        if not isinstance(batch_predictions, Sequence):
            raise ValidationError('Deepchecks requires semantic segmentation predictions to be a sequence with an entry'
                                  ' for each sample')
        if len(batch_predictions) == 0:
            raise ValidationError('Deepchecks requires semantic segmentation predictions to be a non-empty sequence')
        for prediction in batch_predictions:
            if not isinstance(prediction, torch.Tensor):
                raise ValidationError(
                    'Deepchecks requires semantic segmentation predictions to be of type torch.Tensor')
            if len(prediction.shape) != 3:
                raise ValidationError('Deepchecks requires semantic segmentation predictions to be a 3D tensor, but got'
                                      f'a tensor with {len(prediction.shape)} dimensions')
            if n_classes and prediction.shape[0] != n_classes:
                raise ValidationError(f'Deepchecks requires semantic segmentation predictions to have {n_classes} '
                                      'classes')
            if abs(float(prediction[:, 0, 0].sum(dim=0)) - 1) > eps:
                raise ValidationError('Deepchecks requires semantic segmentation predictions to be a probability '
                                      'distribution and sum to 1 for each row')
