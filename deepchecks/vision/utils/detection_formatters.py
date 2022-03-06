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
from itertools import chain
from typing import Union, Callable, Sequence, List, Tuple, Iterable

import numpy as np
import torch
from PIL.Image import Image

from deepchecks.core.errors import DeepchecksValueError
from .base_formatters import BaseLabelFormatter, BasePredictionFormatter


__all__ = ['DetectionLabelFormatter', 'DetectionPredictionFormatter']


class DetectionLabelFormatter(BaseLabelFormatter):
    """
    Class for encoding the detection annotations to the required format.

    Parameters
    ----------
    label_formatter : Callable
        Function that takes in a batch from DataLoader and returns only the encoded labels in the following format:
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
    ... def yolo_to_coco(batch):
    ...     return [torch.stack(
    ...             [torch.cat((bbox[1:3], bbox[4:] - bbox[1:3], bbox[0]), dim=0)
    ...                 for bbox in image])
    ...             for image in batch[1]]
    ...
    ...
    ... label_formatter = DetectionLabelFormatter(yolo_to_coco)


    See Also
    --------
    DetectionPredictionFormatter

    """

    label_formatter: Callable

    def __init__(self, label_formatter: Union[str, Callable] = lambda x: x[1]):
        super().__init__(label_formatter)
        if isinstance(label_formatter, str):
            self.label_formatter = lambda batch: convert_batch_of_bboxes(
                zip(*batch),  # batch - expecting to receive tuple[iterable[image], iterable[bboxes]]
                label_formatter
            )
        else:
            self.label_formatter = label_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.label_formatter(*args, **kwargs)

    def get_classes(self, batch_labels):
        """Get a labels batch and return classes inside it."""
        def get_classes_from_single_label(tensor):
            return list(tensor[:, 0].tolist()) if len(tensor) > 0 else []

        return list(chain(*[get_classes_from_single_label(x) for x in batch_labels]))

    def validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch

        Returns
        -------
        Optional[str]
            None if the label is valid, otherwise a string containing the error message.

        """
        labels = self(batch)
        if not isinstance(labels, list):
            raise DeepchecksValueError('Check requires object detection label to be a list with an entry for each '
                                       'sample')
        if len(labels) == 0:
            raise DeepchecksValueError('Check requires object detection label to be a non-empty list')
        if not isinstance(labels[0], (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires object detection label to be a list of torch.Tensor or numpy '
                                       'array')
        if len(labels[0].shape) != 2:
            raise DeepchecksValueError('Check requires object detection label to be a list of 2D tensors')
        if labels[0].shape[1] != 5:
            raise DeepchecksValueError('Check requires object detection label to be a list of 2D tensors, when '
                                       'each row has 5 columns: [class_id, x, y, width, height]')


class DetectionPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the detection prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch from DataLoader and model, and returns the encoded labels in the
        following format:
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
    ...     batch, model, device
    ... ) -> t.List[torch.Tensor]:
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
    ...
    ...
    ... label_formatter = DetectionPredictionFormatter(yolo_wrapper)


    See Also
    --------
    DetectionLabelFormatter

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
            Batch from DataLoader
        model : t.Any
        device : torch.Device
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
        batch_predictions = self(batch, model, device)
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


def verify_bbox_format_notation(notation: str) -> Tuple[bool, List[str]]:
    """Verify and tokenize bbox format notation.

    Parameters
    ----------
    notation : str
        format notation to verify and to tokenize

    Returns
    -------
    Tuple[
        bool,
        List[Literal['label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'ycenter']]
    ]
        first item indicates whether coordinates are normalized or not,
        second represents format of the bbox
    """
    tokens = []
    are_coordinates_normalized = False
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
        elif current.startswith('n') and current_pos == 0:
            are_coordinates_normalized = True
            current = current[1:]
            current_pos = current_pos + 1
        elif current.startswith('n') and (current_pos + 1) == len(notation):
            are_coordinates_normalized = True
            current_pos = current_pos + 1
            break
        else:
            raise ValueError(
                f'Wrong bbox format notation - {notation}. '
                f'Incorrect or unknown sequence of charecters starting from position {current_pos} '
                f'(sequence: ...{notation[current_pos:]}'
            )

    received_combination = Counter(tokens)
    allowed_combinations = (
        {'l': 1, 'xy': 2},
        {'l': 1, 'xy': 1, 'wh': 1},
        {'l': 1, 'cxcy': 1, 'wh': 1}
    )

    if sum(c == received_combination for c in allowed_combinations) != 1:
        raise ValueError(
            f'Incorrect bbox format notation - {notation}.\n'
            'Only next combinations of elements are allowed:\n'
            '+ lxyxy (label, upper-left corner, bottom-right corner)\n'
            '+ lxywh (label, upper-left corner, bbox width and height)\n'
            '+ lcxcywh (label, bbox center, bbox width and height)\n'
            '+ lcxcywhn (label, normalized bbox center, bbox width and height)\n\n'
            ''
            'Note:\n'
            '- notation elements (l, xy, cxcy, wh) can be placed in any order '
            'but only above combinations of elements are allowed\n'
            '- "n" at the begining or at the ned of the notation indicates '
            'normalized coordinates\n'
        )

    normalized_tokens = []

    for t in tokens:
        if t == 'l':
            normalized_tokens.append('label')
        elif t == 'wh':
            normalized_tokens.extend(('width', 'height'))
        elif t == 'cxcy':
            normalized_tokens.extend(('xcenter', 'ycenter'))
        elif t == 'xy':
            if 'xmin' not in normalized_tokens and 'ymin' not in normalized_tokens:
                normalized_tokens.extend(('xmin', 'ymin'))
            else:
                normalized_tokens.extend(('xmax', 'ymax'))
        else:
            raise RuntimeError('Internal Error! Unreachable part of code reached')

    return are_coordinates_normalized, normalized_tokens


_BatchOfSamples = Iterable[
    Tuple[
        Union[Image, np.ndarray, torch.Tensor],  # images
        Sequence[Sequence[Union[int, float]]]  # bboxes
    ]
]


def convert_batch_of_bboxes(
    batch: _BatchOfSamples,
    notation: str,
    device: Union[str, torch.device, None] = None
) -> List[torch.Tensor]:
    """Convert batch of bboxes to the required format.

    Parameters
    ----------
    batch : tuple like object with two items - list if images, list of bboxes
        batch of images and bboxes
    notation : str
        bboxes format notation
    device : Union[str, torch.device, None], default: None
        device for use

    Returns
    -------
    List[torch.Tensor]
        list of transformed bboxes
    """
    are_coordinates_normalized, notation_tokens = verify_bbox_format_notation(notation)
    output = []

    for image, bboxes in batch:
        if len(bboxes) == 0:
            # image does not have bboxes
            output.append(torch.tensor([]))
            continue

        if are_coordinates_normalized is False:
            image_height = None
            image_width = None
        elif isinstance(image, Image):
            image_height, image_width = image.height, image.width
        elif isinstance(image, (np.ndarray, torch.Tensor)):
            image_height, image_width, *_ = image.shape
        else:
            raise TypeError(
                'Do not know how to take dimension sizes of '
                f'object of type - {type(image)}'
            )

        r = []
        for bbox in bboxes:
            if len(bbox) < 5:
                raise ValueError('incorrect bbox')  # TODO: better message
            else:
                r.append(_convert_bbox(
                    bbox,
                    notation_tokens,
                    device=device,
                    image_width=image_width,
                    image_height=image_height,
                ))
        output.append(torch.stack(r, dim=0))

    return output


def convert_bbox(
    bbox: Sequence[Union[int, float]],
    notation: str,
    image_width: Union[int, float, None] = None,
    image_height: Union[int, float, None] = None,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """Convert bbox to the required format.

    Parameters
    ----------
    bbox : Sequence[Sequence[Union[int, float]]]
        bbox to transform
    notation : str
        bboxes format notation
    image_width : Union[int, float, None], default: None
        width of the image to denormalize bbox coordinates
    image_height : Union[int, float, None], default: None
        height of the image to denormalize bbox coordinates
    device : Union[str, torch.device, None], default: None
        device for use

    Returns
    -------
    torch.Tensor
        bbox transformed to the required by deepchecks format
    """
    if len(bbox) < 5:
        raise ValueError('incorrect bbox')  # TODO: better message

    are_coordinates_normalized, notation_tokens = verify_bbox_format_notation(notation)

    if (
        are_coordinates_normalized is True
        and (image_height is None or image_width is None)
    ):
        raise ValueError(
            'bbox format notation indicates that coordinates of the bbox '
            'are normalized but \'image_height\' and \'image_width\' parameters '
            'were not provided. Please pass image height and width parameters '
            'or remove \'n\' element from the format notation.'
        )

    if (
        are_coordinates_normalized is False
        and (image_height is not None or image_width is not None)
    ):
        raise ValueError(
            'bbox format notation indicates that coordinates of the bbox '
            'are not normalized but \'image_height\' and \'image_width\' were provided. '
            'Those parameters are redundant in the case when bbox coordinates are not '
            'normalized. Please remove those parameters or add \'n\' element to the format '
            'notation to indicate that coordinates are indeed normalized.'
        )

    return _convert_bbox(
        bbox,
        notation_tokens,
        image_width,
        image_height,
        device,
    )


def _convert_bbox(
    bbox: Sequence[Union[int, float]],
    notation_tokens: List[str],
    image_width: Union[int, float, None] = None,
    image_height: Union[int, float, None] = None,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    assert \
        (image_width is not None and image_height is not None) \
        or (image_width is None and image_height is None)

    data = dict(zip(notation_tokens, bbox[:5]))

    if 'xcenter' in data and 'ycenter' in data:
        if image_width is not None and image_height is not None:
            xcenter, ycenter = data['xcenter'] * image_width, data['ycenter'] * image_height
        else:
            xcenter, ycenter = data['xcenter'], data['ycenter']
        return torch.tensor([
            data['label'],
            xcenter - (data['width'] / 2),
            ycenter - (data['height'] / 2),
            data['width'],
            data['height'],
        ], device=device)

    elif 'height' in data and 'width' in data:
        if image_width is not None and image_height is not None:
            xmin, ymin = data['xmin'] * image_width, data['ymin'] * image_height
        else:
            xmin, ymin = data['xmin'], data['ymin']
        return torch.tensor([
            data['label'],
            xmin,
            ymin,
            data['width'],
            data['height'],
        ], device=device)

    else:
        if image_width is not None and image_height is not None:
            xmin, ymin = data['xmin'] * image_width, data['ymin'] * image_height
            xmax, ymax = data['xmax'] * image_width, data['ymax'] * image_height
        else:
            xmin, ymin = data['xmin'], data['ymin']
            xmax, ymax = data['xmax'], data['ymax']
        return torch.tensor([
            data['label'],
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin,
        ], device=device)
