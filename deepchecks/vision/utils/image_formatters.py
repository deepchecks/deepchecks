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
"""Module containing the image formatter class for the vision module."""
from typing import Tuple, List
import cv2
import numpy as np

__all__ = ['IMAGE_PROPERTIES',
           'aspect_ratio',
           'area',
           'brightness',
           'contrast',
           'normalized_red_mean',
           'normalized_blue_mean',
           'normalized_green_mean',
           'get_size',
           'get_dimension']


IMAGE_PROPERTIES = frozenset((
    'aspect_ratio',
    'area',
    'brightness',
    'normalized_red_mean',
    'normalized_green_mean',
    'normalized_blue_mean',
))


def aspect_ratio(batch: List[np.array]) -> List[float]:
    """Return list of floats of image height to width ratio."""
    return [x[0] / x[1] for x in _sizes(batch)]


def area(batch: List[np.array]) -> List[int]:
    """Return list of integers of image areas (height multiplied by width)."""
    return [np.prod(get_size(img)) for img in batch]


def brightness(batch: List[np.array]) -> List[float]:
    """Calculate brightness on each image in the batch."""
    if _is_grayscale(batch) is True:
        return [img.mean() for img in batch]
    else:
        return [(0.299*img[:, 0] + 0.587*img[:, 1] + 0.114 * img[:, 2]).mean() for img in batch]


def contrast(batch: List[np.array]) -> List[float]:
    """Return constrast of image."""
    raise NotImplementedError('Not yet implemented')  # TODO


def normalized_red_mean(batch: List[np.array]) -> List[float]:
    """Return the normalized mean of the red channel."""
    return [x[0] for x in _normalized_rgb_mean(batch)]


def normalized_green_mean(batch: List[np.array]) -> List[float]:
    """Return the normalized mean of the green channel."""
    return [x[1] for x in _normalized_rgb_mean(batch)]


def normalized_blue_mean(batch: List[np.array]) -> List[float]:
    """Return the normalized mean of the blue channel."""
    return [x[2] for x in _normalized_rgb_mean(batch)]


def _sizes(batch: List[np.array]):
    """Return list of tuples of image height and width."""
    return [get_size(img) for img in batch]


def _normalized_rgb_mean(batch: List[np.array]) -> List[Tuple[float, float, float]]:
    """Calculate normalized mean for each channel (rgb) in image.

    The normalized mean of each channel is calculated by first normalizing the image's pixels (meaning, each color
    is normalized to its relevant intensity, by dividing the color intensity by the other colors). Then, the mean
    for each image channel is calculated.

    Parameters
    ----------
    batch: List[np.array]
        A list of arrays, each arrays represents an image in the required deepchecks format.
    sample_size_for_image_properties: int
        The number of pixels to sample from each image.

    Returns
    -------
    List[np.array]:
        List of 3-dimensional arrays, each dimension is the normalized mean of the color channel. An array is
        returned for each image.
    """
    if _is_grayscale(batch) is True:
        return [(None, None, None)] * len(batch)

    return [cv2.mean(
        cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
                           for img in batch]


def _is_grayscale(batch):
    return get_dimension(batch[0]) == 1


def get_size(img) -> Tuple[int, int]:
    """Get size of image as (height, width) tuple."""
    return img.shape[0], img.shape[1]


def get_dimension(img) -> int:
    """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
    return img.shape[2]
