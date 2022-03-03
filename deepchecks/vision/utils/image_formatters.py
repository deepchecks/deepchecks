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

import numpy as np

from deepchecks.core.errors import DeepchecksValueError

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


def brightness(batch: List[np.array], sample_size_for_image_properties=10000) -> List[float]:
    """Calculate brightness on each image in the batch."""
    if _is_grayscale(batch) is True:
        return [img.mean() for img in batch]
    else:
        flattened_batch = _flatten_batch(batch, sample_size_for_image_properties)
        return [(0.299*img[:, 0] + 0.587*img[:, 1] + 0.114 * img[:, 2]).mean() for img in flattened_batch]


def contrast(batch: List[np.array]) -> List[float]:
    """Return constrast of image."""
    raise NotImplementedError('Not yet implemented')  # TODO


def normalized_red_mean(batch: List[np.array], sample_size_for_image_properties=10000) -> List[float]:
    """Return the normalized mean of the red channel."""
    return [x[0] for x in _normalized_rgb_mean(batch, sample_size_for_image_properties)]


def normalized_green_mean(batch: List[np.array], sample_size_for_image_properties=10000) -> List[float]:
    """Return the normalized mean of the green channel."""
    return [x[1] for x in _normalized_rgb_mean(batch, sample_size_for_image_properties)]


def normalized_blue_mean(batch: List[np.array], sample_size_for_image_properties=10000) -> List[float]:
    """Return the normalized mean of the blue channel."""
    return [x[2] for x in _normalized_rgb_mean(batch, sample_size_for_image_properties)]


def _sizes(batch: List[np.array]):
    """Return list of tuples of image height and width."""
    return [get_size(img) for img in batch]


def _normalized_rgb_mean(batch: List[np.array], sample_size_for_image_properties) -> List[Tuple[float, float, float]]:
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

    flattened_batch = _flatten_batch(batch, sample_size_for_image_properties)
    # TODO: Check for faster implementations than pixel by pixel
    normalized_images = [np.array([_normalize_colors_in_pixel(pxl) for pxl in img]) for img in flattened_batch]
    return [img.mean(axis=0) for img in normalized_images]


def _normalize_colors_in_pixel(pixel):
    pxl_sum = pixel.sum()
    return np.array([pixel[i] / pxl_sum if pxl_sum else 0 for i in range(3)])


def _is_grayscale(batch):
    return get_dimension(batch[0]) == 1


def _sample_images_in_batch(flattened_batch, sample_size_for_image_properties):
    if sample_size_for_image_properties is None:
        raise RuntimeError(
            'function sample_images_in_batch should not be called if sample_size_for_image_properties is None')
    ret_batch = []
    np.random.seed(len(flattened_batch))
    for img in flattened_batch:
        if img.shape[0] <= sample_size_for_image_properties:
            ret_batch.append(img)
        else:
            indexes = np.random.randint(0, img.shape[0], sample_size_for_image_properties)
            sampled_img = np.array([img[i, :] for i in indexes])
            ret_batch.append(sampled_img)

    return ret_batch


def _flatten_batch(batch, sample_size_for_image_properties):
    if _is_grayscale(batch) is True:
        raise DeepchecksValueError('function _flatten_batch cannot run on 1-dimensional image (grayscale)')
    flattened_imgs = [img.reshape([img.shape[0] * img.shape[1], 3]) for img in batch]
    if sample_size_for_image_properties is not None:
        return _sample_images_in_batch(flattened_imgs, sample_size_for_image_properties)
    else:
        return flattened_imgs


def get_size(img) -> Tuple[int, int]:
    """Get size of image as (height, width) tuple."""
    return img.shape[0], img.shape[1]


def get_dimension(img) -> int:
    """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
    return img.shape[2]
