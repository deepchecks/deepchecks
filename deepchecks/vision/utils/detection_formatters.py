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
"""Module for defining detection encoders."""
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

__all__ = ['verify_bbox_format_notation', 'convert_batch_of_bboxes', 'convert_bbox', 'DEFAULT_PREDICTION_FORMAT']


DEFAULT_PREDICTION_FORMAT = 'xywhsl'


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
        List[Literal['label', 'score', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'ycenter']]
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
        elif current.startswith('s'):
            tokens.append('s')
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
    allowed_combinations = [
        {'l': 1, 'xy': 2},
        {'l': 1, 'xy': 1, 'wh': 1},
        {'l': 1, 'cxcy': 1, 'wh': 1}
    ]
    # All allowed combinations are also allowed with or without score to support both label and prediction
    allowed_combinations += [{**c, 's': 1} for c in allowed_combinations]

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
        elif t == 's':
            normalized_tokens.append('score')
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
        Union[Image, np.ndarray, torch.Tensor],  # image
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
    batch : iterable of tuple like object with two items - image, list of bboxes
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
    device: Union[str, torch.device, None] = None,
    _strict: bool = True  # pylint: disable=invalid-name
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
        if _strict is True:
            raise ValueError(
                'bbox format notation indicates that coordinates of the bbox '
                'are not normalized but \'image_height\' and \'image_width\' were provided. '
                'Those parameters are redundant in the case when bbox coordinates are not '
                'normalized. Please remove those parameters or add \'n\' element to the format '
                'notation to indicate that coordinates are indeed normalized.'
            )
        else:
            image_height = None
            image_width = None

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

    data = dict(zip(notation_tokens, bbox))

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
