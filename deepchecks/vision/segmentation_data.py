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
from typing import List, Sequence, Tuple, Any

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
    def batch_to_labels(self, batch) -> Tuple[List[List[Any]], List[torch.Tensor]]:
        """Extract the labels from a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        Tuple[List[List[Any]], List[torch.Tensor]]
            The labels extracted from the batch. The labels should be a tuple of 2 lists:
            1st list: list of length N containing lists of detected classes (each list references a single image)
            2nd list: list of length N containing torch.Tensor of shape (k, img_width, img_height), where k is the
                number of detections per that image.
            See the notes for more info.

        Examples #TODO: Improve code of example
        --------
        >>> import torch
        ...
        ...
        ... def batch_to_labels(self, batch):
        ...     # each image's label is of format torch.Tensor, where the tensor is of correct size, but each pixel
        ...     # represents a different class_id (or none).
        ...     # We would like to convert to a format where the function returns a tuple of 2 lists: one for class_ids
        ...     # and one for tensors of boolean images:
        ...     class_ids: list = self._class_ids
        ...     classes_list = []
        ...     tensors_list = []
        ...     for label in batch[1]:
        ...         image_classes = []
        ...         image_segments = []
        ...         for class_id in class_ids:
        ...             detection_tensor = label == class_id
        ...             if detection_tensor.sum() == 0:
        ...                 continue
        ...             else:
        ...                 image_classes.append(class_id)
        ...                 image_segments.append(detection_tensor)
        ...         tensors_list.append(image_classes)
        ...         classes_list.append(torch.stack(image_segments))


        Notes
        -----
        The accepted label format for is a tuple of 2 lists:
            1st list: list of length N containing lists of detected classes (each list references a single image)
            2nd list: list of length N containing torch.Tensor of shape (k, img_width, img_height), where k is the
                number of detections per that image, and img_width and img_height are the dimensions of the image
                (regardless of addional channels).
                Practically, this tensor is k boolean images, each one correlates to a detection and class, and the
                values themselves reference each pixel in the original image and whether that object class was detected
                in them or not.
        """
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Sequence[torch.Tensor]:
        """Return the predictions of the model on a batch of data. #TODO: Define and refactor docstring

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
        """Get a labels batch and return classes inside it.""" #TODO: Refactor

        def get_classes_from_single_label(tensor: torch.Tensor):
            return list(tensor[:, 0].type(torch.IntTensor).tolist()) if len(tensor) > 0 else []

        return [get_classes_from_single_label(x) for x in batch_labels]

    def validate_label(self, batch):
        """ #TODO: Refactor
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
        if not len(labels) == 2 or not isinstance(labels[0], list) or not isinstance(labels[1], list):
            raise ValidationError('Deepchecks requires semantic segmentation label to be a tuple of 2 lists, one for '
                                  'class ids and one for boolean segment maps')
        if len(labels[0]) == 0:
            raise ValidationError('Deepchecks requires semantic segmentation label to be a non-empty list')
        if not isinstance(labels[0][0], list):
            raise ValidationError('Deepchecks requires semantic segmentation label 1st list to be a list of lists')
        if not isinstance(labels[1][0], torch.Tensor):
            raise ValidationError('Deepchecks requires semantic segmentation label 2nd list to be a list of '
                                  'torch.Tensor')
        sample_idx = 0
        # Find a non empty tensor to validate
        # while labels[0][sample_idx].shape[0] == 0:
        #     sample_idx += 1
        #     if sample_idx == len(labels):
        #         return  # No labels to validate
        # if len(labels[1][sample_idx][0].shape) != 2:
        #     raise ValidationError('Check requires object detection label to be a list of 2D tensors')
        # if labels[sample_idx].shape[1] != 5:
        #     raise ValidationError('Check requires object detection label to be a list of 2D tensors, when '
        #                           'each row has 5 columns: [class_id, x, y, width, height]')
        # if torch.min(labels[sample_idx]) < 0:
        #     raise ValidationError('Found one of coordinates to be negative, check requires object detection '
        #                           'bounding box coordinates to be of format [class_id, x, y, width, height].')
        # if torch.max(labels[sample_idx][:, 0] % 1) > 0:
        #     raise ValidationError('Class_id must be a positive integer. Object detection labels per image should '
        #                           'be a Bx5 tensor of format [class_id, x, y, width, height].')

    @staticmethod
    def validate_infered_batch_predictions(batch_predictions):
        """ #TODO: Refactor
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
        # if not isinstance(batch_predictions, Sequence):
        #     raise ValidationError('Check requires detection predictions to be a sequence with an entry for each'
        #                           ' sample')
        # if len(batch_predictions) == 0:
        #     raise ValidationError('Check requires detection predictions to be a non-empty sequence')
        # if not isinstance(batch_predictions[0], torch.Tensor):
        #     raise ValidationError('Check requires detection predictions to be a sequence of torch.Tensor')
        # sample_idx = 0
        # # Find a non empty tensor to validate
        # while batch_predictions[sample_idx].shape[0] == 0:
        #     sample_idx += 1
        #     if sample_idx == len(batch_predictions):
        #         return  # No predictions to validate
        # if len(batch_predictions[sample_idx].shape) != 2:
        #     raise ValidationError('Check requires detection predictions to be a sequence of 2D tensors')
        # if batch_predictions[sample_idx].shape[1] != 6:
        #     raise ValidationError('Check requires detection predictions to be a sequence of 2D tensors, when '
        #                           'each row has 6 columns: [x, y, width, height, class_probability, class_id]')
        # if torch.min(batch_predictions[sample_idx]) < 0:
        #     raise ValidationError('Found one of coordinates to be negative, Check requires object detection '
        #                           'bounding box predictions to be of format [x, y, width, height, confidence,'
        #                           ' class_id]. ')
        # if torch.min(batch_predictions[sample_idx][:, 4]) < 0 or torch.max(batch_predictions[sample_idx][:, 4]) > 1:
        #     raise ValidationError('Confidence must be between 0 and 1. Object detection predictions per image '
        #                           'should be a Bx6 tensor of format [x, y, width, height, confidence, class_id].')
        # if torch.max(batch_predictions[sample_idx][:, 5] % 1) > 0:
        #     raise ValidationError('Class_id must be a positive integer. Object detection predictions per image '
        #                           'should be a Bx6 tensor of format [x, y, width, height, confidence, class_id].')
