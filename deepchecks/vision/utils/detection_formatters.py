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
from typing import Union, Callable, Optional, Sequence, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksValueError
from .base_formatters import BaseLabelFormatter, BasePredictionFormatter


__all__ = ['DetectionLabelFormatter', 'DetectionPredictionFormatter']


class DetectionLabelFormatter(BaseLabelFormatter):
    """
    Class for encoding the detection annotations to the required format.

    Parameters
    ----------
    label_formatter : Callable
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        List of length N containing tensors of shape (M, 5), where N is the number of samples,
        M is the number of bounding boxes in the sample and each bounding box is represented by 5 values: **(class_id,
        x, y, w, h)**. x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
        the width and height of the bounding box (in pixels) and class_id is the class id.

    Examples
    --------
    >>> import torch
    ... from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
    ...
    ...
    ... def yolo_to_coco(input_batch_from_loader):
    ...     return [torch.stack(
    ...             [torch.cat((bbox[1:3], bbox[4:] - bbox[1:3], bbox[0]), dim=0)
    ...                 for bbox in image])
    ...             for image in input_batch_from_loader]
    ...
    ...
    ... label_formatter = DetectionLabelFormatter(yolo_to_coco)


    See Also
    --------
    DetectionPredictionFormatter

    """

    label_formatter: Callable

    def __init__(
        self,
        label_formatter: Union[str, Callable] = lambda x: x
    ):
        super().__init__(label_formatter)
        if isinstance(label_formatter, str):
            self.label_formatter = lambda batch: convert_batch_of_bboxes(batch, label_formatter)
        else:
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
            Counter of the number of samples per class.
        """
        counter = Counter()
        for batch in data_loader:
            list_of_arrays = self(batch[1])
            class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
            counter.update(class_list)
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
            return 'Check requires dataset to have a label'

        label_batch = self(batch[1])
        if not isinstance(label_batch, list):
            return 'Check requires object detection label to be a list with an entry for each sample'
        if len(label_batch) == 0:
            return 'Check requires object detection label to be a non-empty list'
        if not isinstance(label_batch[0], (torch.Tensor, np.ndarray)):
            return 'Check requires object detection label to be a list of torch.Tensor or numpy array'
        if len(label_batch[0].shape) != 2:
            return 'Check requires object detection label to be a list of 2D tensors'
        if label_batch[0].shape[1] != 5:
            return 'Check requires object detection label to be a list of 2D tensors, when ' \
                   'each row has 5 columns: [class_id, x, y, width, height]'


class DetectionPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the detection prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch of predictions and returns the encoded labels in the following format:
        List of length N containing tensors of shape (B, 6), where N is the number of images,
        B is the number of bounding boxes detected in the sample and each bounding box is represented by 6 values:
        **[x, y, w, h, confidence, class_id]**. x and y are the coordinates (in pixels) of the upper left corner of the
        bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the confidence of
        the model and class_id is the class id.

    Examples
    --------
    >>> import torch
    ... import typing as t
    ... from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
    ...
    ...
    ... def yolo_wrapper(
    ...     predictions: 'ultralytics.models.common.Detections'  # noqa: F821
    ... ) -> t.List[torch.Tensor]:
    ...     return_list = []
    ...
    ...     # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
    ...     for single_image_tensor in predictions.pred:
    ...         pred_modified = torch.clone(single_image_tensor)
    ...         pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
    ...         pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
    ...         return_list.append(pred_modified)
    ...
    ...     return return_list
    ...
    ...
    ... label_formatter = DetectionPredictionFormatter(yolo_wrapper)


    See Also
    --------
    DetectionLabelFormatter

    """

    def __init__(self, prediction_formatter: Callable = lambda x: x):
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
        if not isinstance(batch_predictions, list):
            raise DeepchecksValueError('Check requires detection predictions to be a list with an entry for each'
                                       ' sample')
        if len(batch_predictions) == 0:
            raise DeepchecksValueError('Check requires detection predictions to be a non-empty list')
        if not isinstance(batch_predictions[0], (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires detection predictions to be a list of torch.Tensor or'
                                       ' numpy array')
        if len(batch_predictions[0].shape) != 2:
            raise DeepchecksValueError('Check requires detection predictions to be a list of 2D tensors')
        if batch_predictions[0].shape[1] != 6:
            raise DeepchecksValueError('Check requires detection predictions to be a list of 2D tensors, when '
                                       'each row has 6 columns: [x, y, width, height, class_probability, class_id]')


def verify_bbox_format_notation(notation: str) -> List[str]:
    """Verify and tokenize bbox format notation.

    Parameters
    ----------
    notation : str
        format notation to verify and to tokenize

    Returns
    -------
    List[Literal['label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'ycenter']]
    """
    tokens = []
    current = notation = notation.strip().lower()
    current_pos = 0

    while current:
        if current.startswith('l'):
            tokens.append('l')
            current = current[1:]
            current_pos = current_pos + 1
        elif current.startswith('wh'):
            tokens.append('wh')
            current = current[2:]
            current_pos = current_pos + 2
        elif current.startswith('xy'):
            tokens.append('xy')
            current = current[2:]
            current_pos = current_pos + 2
        elif current.startswith('cxcy'):
            tokens.append('cxcy')
            current = current[4:]
            current_pos = current_pos + 4
        else:
            raise ValueError(
                f'Incorrect bbox format notation - {notation}. '
                f'Unknown sequence of charecters starting from position {current_pos} '
                f'(sequence: ...{notation[current_pos:]}'
            )

    received_combination = Counter(tokens)
    allowed_combinations = (
        {'l': 1, 'xy': 2},
        {'l': 1, 'xy': 1, 'wh': 1},
        {'l': 1, 'cxcy': 1, 'wh': 1}
    )

    if not any(c == received_combination for c in allowed_combinations):
        raise ValueError(
            f'Incorrect bbox format notation - {notation}.\n'
            'Only next combinations of elements are allowed:\n'
            '+ lxyxy (label, upper-left corner, bottom-right corner)\n'
            '+ lxywh (label, upper-left corner, bbox width and height)\n'
            '+ lcxcywh (label, bbox center, bbox width and height)\n\n'
            ''
            'Note:\n'
            '- notation elements (l, xy, cxcy, wh) can be placed in any order '
            'but only above combinations of elements are allowed.'
        )

    normilized_tokens = []

    for t in tokens:
        if t == 'l':
            normilized_tokens.append('label')
        elif t == 'wh':
            normilized_tokens.extend(('width', 'height'))
        elif t == 'cxcy':
            normilized_tokens.extend(('xcenter', 'ycenter'))
        elif t == 'xy':
            if 'xmin' not in normilized_tokens and 'ymin' not in normilized_tokens:
                normilized_tokens.extend(('xmin', 'ymin'))
            else:
                normilized_tokens.extend(('xmax', 'ymax'))
        else:
            raise RuntimeError('Internal Error! Unreachable part of code reached')

    return normilized_tokens


def convert_batch_of_bboxes(
    batch: Sequence[Sequence[Sequence[Union[int, float]]]],
    notation: str,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """Convert batch of bboxes to the required format.

    Parameters
    ----------
    bboxes : Sequence[Sequence[Union[int, float]]]
        batch of bboxes to transform
    notation : str
        bboxes format notation
    device : Union[str, torch.device, None], default: None
        device for use

    Returns
    -------
    torch.Tensor
        tensor of transformed samples of bboxes
    """
    notation_tokens = verify_bbox_format_notation(notation)
    output = []
    for sample in batch:
        r = []
        for bbox in sample:
            if len(bbox) < 5:
                raise ValueError('incorrect bbox')  # TODO: better message
            else:
                r.append(_convert_bbox(bbox, notation_tokens, device))
        output.append(r)
    return torch.tensor(output)


def convert_bbox(
    bbox: Sequence[Union[int, float]],
    notation: str,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """Convert bbox to the required format.

    Parameters
    ----------
    bboxes : Sequence[Sequence[Union[int, float]]]
        batch of bboxes to transform
    notation : str
        bboxes format notation
    device : Union[str, torch.device, None], default: None
        device for use

    Returns
    -------
    torch.Tensor
        bbox transformed to the required by deepchecks format
    """
    if len(bbox) < 5:
        raise ValueError('incorrect bbox')  # TODO: better message
    notation_tokens = verify_bbox_format_notation(notation)
    return _convert_bbox(bbox, notation_tokens, device)


def _convert_bbox(
    bbox: Sequence[Union[int, float]],
    notation_tokens: List[str],
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    data = dict(zip(notation_tokens, bbox[:5]))
    if 'xcenter' in data and 'ycenter' in data:
        return torch.tensor([
            data['label'],
            data['xcenter'] - (data['width'] / 2),
            data['ycenter'] - (data['height'] / 2),
            data['width'],
            data['height'],
        ], device=device)
    elif 'height' in data and 'width' in data:
        return torch.tensor([
            data['label'],
            data['xmin'],
            data['ymin'],
            data['width'],
            data['height'],
        ], device=device)
    else:
        return torch.tensor([
            data['label'],
            data['xmin'],
            data['ymin'],
            data['xmax'] - data['xmin'],
            data['ymax'] - data['ymin'],
        ], device=device)
