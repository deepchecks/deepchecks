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
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from skimage.color import rgb2gray

from deepchecks.core.errors import DeepchecksValueError

__all__ = ['default_image_properties',
           'aspect_ratio',
           'area',
           'brightness',
           'rms_contrast',
           'mean_red_relative_intensity',
           'mean_blue_relative_intensity',
           'mean_green_relative_intensity',
           'get_size',
           'get_dimension',
           'validate_properties',
           'get_column_type']


def aspect_ratio(batch: List[np.ndarray]) -> List[float]:
    """Return list of floats of image height to width ratio."""
    return [x[0] / x[1] for x in _sizes(batch)]


def area(batch: List[np.ndarray]) -> List[int]:
    """Return list of integers of image areas (height multiplied by width)."""
    return [np.prod(get_size(img)) for img in batch]


def brightness(batch: List[np.ndarray]) -> List[float]:
    """Calculate brightness on each image in the batch."""
    return [img.mean() if _is_grayscale(img) else rgb2gray(img).mean()
            for img in batch]


def rms_contrast(batch: List[np.array]) -> List[float]:
    """Return RMS contrast of image."""
    return [img.std() if _is_grayscale(img) else rgb2gray(img).std()
            for img in batch]


def mean_red_relative_intensity(batch: List[np.ndarray]) -> List[float]:
    """Return the mean of the red channel relative intensity."""
    return [x[0] for x in _rgb_relative_intensity_mean(batch)]


def mean_green_relative_intensity(batch: List[np.ndarray]) -> List[float]:
    """Return the mean of the green channel relative intensity."""
    return [x[1] for x in _rgb_relative_intensity_mean(batch)]


def mean_blue_relative_intensity(batch: List[np.ndarray]) -> List[float]:
    """Return the mean of the blue channel relative intensity."""
    return [x[2] for x in _rgb_relative_intensity_mean(batch)]


def _sizes(batch: List[np.ndarray]):
    """Return list of tuples of image height and width."""
    return [get_size(img) for img in batch]


def _rgb_relative_intensity_mean(batch: List[np.ndarray]) -> List[Tuple[float, float, float]]:
    """Calculate normalized mean for each channel (rgb) in image.

    The normalized mean of each channel is calculated by first normalizing the image's pixels (meaning, each color
    is normalized to its relevant intensity, by dividing the color intensity by the other colors). Then, the mean
    for each image channel is calculated.

    Parameters
    ----------
    batch: List[np.ndarray]
        A list of arrays, each arrays represents an image in the required deepchecks format.

    Returns
    -------
    List[np.ndarray]:
        List of 3-dimensional arrays, each dimension is the normalized mean of the color channel. An array is
        returned for each image.
    """
    return [_normalize_pixelwise(img).mean(axis=(1, 2)) if not _is_grayscale(img) else (None, None, None)
            for img in batch]


def _normalize_pixelwise(img: np.ndarray) -> np.ndarray:
    """Normalize the pixel values of an image.

    Parameters
    ----------
    img: np.ndarray
        The image to normalize.

    Returns
    -------
    np.ndarray
        The normalized image.
    """
    s = img.sum(axis=2)
    return np.array([np.divide(img[:, :, i], s, out=np.zeros_like(img[:, :, i], dtype='float64'), where=s != 0)
                     for i in range(3)])


def _is_grayscale(img):
    return get_dimension(img) == 1


def get_size(img) -> Tuple[int, int]:
    """Get size of image as (height, width) tuple."""
    return img.shape[0], img.shape[1]


def get_dimension(img) -> int:
    """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
    return img.shape[2]


default_image_properties = [
    {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'numerical'},
    {'name': 'Area', 'method': area, 'output_type': 'numerical'},
    {'name': 'Brightness', 'method': brightness, 'output_type': 'numerical'},
    {'name': 'RMS Contrast', 'method': rms_contrast, 'output_type': 'numerical'},
    {'name': 'Mean Red Relative Intensity', 'method': mean_red_relative_intensity, 'output_type': 'numerical'},
    {'name': 'Mean Green Relative Intensity', 'method': mean_green_relative_intensity, 'output_type': 'numerical'},
    {'name': 'Mean Blue Relative Intensity', 'method': mean_blue_relative_intensity, 'output_type': 'numerical'}
]


def validate_properties(properties: List[Dict[str, Any]]):
    """Validate structure of measurements."""
    if not isinstance(properties, list):
        raise DeepchecksValueError(
            'Expected properties to be a list, '
            f'instead got {type(properties).__name__}'
        )

    if len(properties) == 0:
        raise DeepchecksValueError('Properties list can\'t be empty')

    expected_keys = ('name', 'method', 'output_type')
    deprecated_output_types = ('discrete', 'continuous')
    output_types = ('categorical', 'numerical')

    list_of_warnings = []
    errors = []

    for index, image_property in enumerate(properties):

        if not isinstance(image_property, dict):
            errors.append(
                f'Item #{index}: property must be of type dict, '
                f'and include keys {expected_keys}. Instead got {type(image_property).__name__}'
            )
            continue

        property_name = image_property.get('name') or f'#{index}'
        difference = sorted(set(expected_keys).difference(set(image_property.keys())))

        if len(difference) > 0:
            errors.append(
                f'Property {property_name}: dictionary must include keys {expected_keys}. '
                f'Next keys are missed {difference}'
            )
            continue

        property_output_type = image_property['output_type']

        if property_output_type in deprecated_output_types:
            list_of_warnings.append(
                f'Property {property_name}: output types {deprecated_output_types} are deprecated, '
                f'use instead {output_types}'
            )
        elif property_output_type not in output_types:
            errors.append(
                f'Property {property_name}: field "output_type" must be one of {output_types}, '
                f'instead got {property_output_type}'
            )

    if len(errors) > 0:
        errors = '\n+ '.join(errors)
        raise DeepchecksValueError(f'List of properties contains next problems:\n+ {errors}')

    if len(list_of_warnings) > 0:
        concatenated_warnings = '\n+ '.join(list_of_warnings)
        warnings.warn(
            f'Property Warnings:\n+ {concatenated_warnings}',
            category=DeprecationWarning
        )

    return properties


def get_column_type(output_type):
    """Get column type to use in drift functions."""
    # TODO: smarter mapping based on data?
    # NOTE/TODO: this function is kept only for backward compatibility, remove it later
    mapper = {
        'continuous': 'numerical',
        'discrete': 'categorical',
        'numerical': 'numerical',
        'categorical': 'categorical'
    }
    return mapper[output_type]
