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
from typing import List, Sequence

import torch

from deepchecks.core.errors import DeepchecksNotImplementedError, ValidationError
from deepchecks.vision.vision_data import TaskType, VisionData


class DetectionData(VisionData):
    """The DetectionData class is used to load and preprocess data for a object detection task.

    It is a subclass of the VisionData class. The DetectionData class is containing additional data and general
    methods intended for easily accessing metadata relevant for validating a computer vision object detection ML models.
    """

    @property
    def task_type(self) -> TaskType:
        """Return the task type (object_detection)."""
        return TaskType.OBJECT_DETECTION

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
            The labels extracted from the batch. The labels should be a list of length N containing tensor of shape
            (B, 5) where N is the number of samples, B is the number of bounding boxes in the sample and each bounding
            box is represented by 5 values. See the notes for more info.

        Examples
        --------
        >>> import torch
        ...
        ...
        ... def batch_to_labels(self, batch):
        ...     # each bbox in the labels is (class_id, x, y, x, y). convert to (class_id, x, y, w, h)
        ...     return [torch.stack(
        ...            [torch.cat((bbox[0], bbox[1:3], bbox[4:] - bbox[1:3]), dim=0)
        ...                for bbox in image])
        ...             for image in batch[1]]

        Notes
        -----
        The accepted label format for is a a list of length N containing tensors of shape (B, 5), where N is the number
        of samples, B is the number of bounding boxes in the sample and each bounding box is represented by 5 values:
        (class_id, x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w
         and h are the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.
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
            tensors of shape (B, 6), where N is the number of images, B is the number of bounding boxes detected in the
            sample and each bounding box is represented by 6 values. See the notes for more info.

        Examples
        --------
        >>> import torch
        ...
        ...
        ... def infer_on_batch(self, batch, model, device):
        ...     # Converts a yolo prediction batch to the accepted xywh format
        ...     return_list = []
        ...
        ...     predictions = model(batch[0])
        ...     # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        ...     for single_image_tensor in predictions.pred:
        ...         pred_modified = torch.clone(single_image_tensor)
        ...         pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
        ...         pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
        ...         return_list.append(pred_modified)
        ...
        ...     return return_list

        Notes
        -----
        The accepted prediction format is a list of length N containing tensors of shape (B, 6), where N is the number
        of images, B is the number of bounding boxes detected in the sample and each bounding box is represented by 6
        values: [x, y, w, h, confidence, class_id]. x and y are the coordinates (in pixels) of the upper left corner
        of the bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the
        confidence of the model and class_id is the class id.
        """
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    def get_classes(self, batch_labels: List[torch.Tensor]):
        """Get a labels batch and return classes inside it."""

        def get_classes_from_single_label(tensor: torch.Tensor):
            return list(tensor[:, 0].type(torch.IntTensor).tolist()) if len(tensor) > 0 else []

        return [get_classes_from_single_label(x) for x in batch_labels]

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
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, list):
            raise ValidationError('Check requires object detection label to be a list with an entry for each '
                                  'sample')
        if len(labels) == 0:
            raise ValidationError('Check requires object detection label to be a non-empty list')
        if not isinstance(labels[0], torch.Tensor):
            raise ValidationError('Check requires object detection label to be a list of torch.Tensor')
        sample_idx = 0
        # Find a non empty tensor to validate
        while labels[sample_idx].shape[0] == 0:
            sample_idx += 1
            if sample_idx == len(labels):
                return  # No labels to validate
        if len(labels[sample_idx].shape) != 2:
            raise ValidationError('Check requires object detection label to be a list of 2D tensors')
        if labels[sample_idx].shape[1] != 5:
            raise ValidationError('Check requires object detection label to be a list of 2D tensors, when '
                                  'each row has 5 columns: [class_id, x, y, width, height]')
        if torch.min(labels[sample_idx]) < 0:
            raise ValidationError('Found one of coordinates to be negative, check requires object detection '
                                  'bounding box coordinates to be of format [class_id, x, y, width, height].')
        if torch.max(labels[sample_idx][:, 0] % 1) > 0:
            raise ValidationError('Class_id must be a positive integer. Object detection labels per image should '
                                  'be a Bx5 tensor of format [class_id, x, y, width, height].')

    @staticmethod
    def validate_infered_batch_predictions(batch_predictions):
        """
        Validate the infered predictions from the batch.

        Parameters
        ----------
        batch_predictions : t.Any
            The infered predictions from the batch

        Raises
        ------
        ValidationError
            If predictions format is invalid
        DeepchecksNotImplementedError
            If infer_on_batch not implemented
        """
        if not isinstance(batch_predictions, Sequence):
            raise ValidationError('Check requires detection predictions to be a sequence with an entry for each'
                                  ' sample')
        if len(batch_predictions) == 0:
            raise ValidationError('Check requires detection predictions to be a non-empty sequence')
        if not isinstance(batch_predictions[0], torch.Tensor):
            raise ValidationError('Check requires detection predictions to be a sequence of torch.Tensor')
        sample_idx = 0
        # Find a non empty tensor to validate
        while batch_predictions[sample_idx].shape[0] == 0:
            sample_idx += 1
            if sample_idx == len(batch_predictions):
                return  # No predictions to validate
        if len(batch_predictions[sample_idx].shape) != 2:
            raise ValidationError('Check requires detection predictions to be a sequence of 2D tensors')
        if batch_predictions[sample_idx].shape[1] != 6:
            raise ValidationError('Check requires detection predictions to be a sequence of 2D tensors, when '
                                  'each row has 6 columns: [x, y, width, height, class_probability, class_id]')
        if torch.min(batch_predictions[sample_idx]) < 0:
            raise ValidationError('Found one of coordinates to be negative, Check requires object detection '
                                  'bounding box predictions to be of format [x, y, width, height, confidence,'
                                  ' class_id]. ')
        if torch.min(batch_predictions[sample_idx][:, 4]) < 0 or torch.max(batch_predictions[sample_idx][:, 4]) > 1:
            raise ValidationError('Confidence must be between 0 and 1. Object detection predictions per image '
                                  'should be a Bx6 tensor of format [x, y, width, height, confidence, class_id].')
        if torch.max(batch_predictions[sample_idx][:, 5] % 1) > 0:
            raise ValidationError('Class_id must be a positive integer. Object detection predictions per image '
                                  'should be a Bx6 tensor of format [x, y, width, height, confidence, class_id].')
