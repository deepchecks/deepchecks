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
"""Module containing the image formatter class for the vision module."""
from typing import Dict, List, Tuple

import numpy as np
from cv2 import CV_64F, Laplacian
from skimage.color import rgb2gray

__all__ = ['default_image_properties',
           'calc_default_image_properties',
           'aspect_ratio',
           'area',
           'brightness',
           'rms_contrast',
           'mean_red_relative_intensity',
           'mean_blue_relative_intensity',
           'mean_green_relative_intensity',
           'get_size',
           'get_dimension']


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


def texture_level(batch: List[np.ndarray]) -> List[float]:
    """Calculate the sharpness of each image in the batch."""
    return [Laplacian(img, CV_64F).var() if _is_grayscale(img) else Laplacian(rgb2gray(img), CV_64F).var()
            for img in batch]


def _sizes(batch: List[np.ndarray]):
    """Return list of tuples of image height and width."""
    return [get_size(img) for img in batch]


def _sizes_array(batch: List[np.ndarray]):
    """Return an array of height and width per image (Nx2)."""
    return np.array(_sizes(batch))


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


def _rgb_relative_intensity_mean_array(batch: List[np.ndarray]) -> np.ndarray:
    """Return the _rgb_relative_intensity_mean result as array."""
    return np.array(_rgb_relative_intensity_mean(batch))


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


def sample_pixels(image: np.ndarray, n_pixels: int):
    """Sample the image to improve runtime, expected image format H,W,C."""
    flat_image = image.reshape((-1, image.shape[-1]))
    if flat_image.shape[0] > n_pixels:
        pixel_idxs = np.random.choice(flat_image.shape[0], n_pixels)
    else:
        pixel_idxs = np.arange(flat_image.shape[0])
    sampled_image = flat_image[pixel_idxs, np.newaxis, :]
    return sampled_image


def calc_default_image_properties(batch: List[np.ndarray], sample_n_pixels: int = 10000) -> Dict[str, list]:
    """Speed up the calculation for the default image properties by sharing common actions."""
    if len(batch) == 0:
        return {}
    results_dict = {}
    sizes_array = _sizes_array(batch)
    results_dict['Aspect Ratio'] = list(sizes_array[:, 0] / sizes_array[:, 1])
    results_dict['Area'] = list(sizes_array[:, 0] * sizes_array[:, 1])

    sampled_images = [sample_pixels(img, sample_n_pixels) for img in batch]

    grayscale_images = [img if _is_grayscale(img) else rgb2gray(img)*255 for img in sampled_images]
    results_dict['Brightness'] = [image.mean() for image in grayscale_images]
    results_dict['RMS Contrast'] = [image.std() for image in grayscale_images]

    rgb_intensities = _rgb_relative_intensity_mean_array(sampled_images)
    results_dict['Mean Red Relative Intensity'] = rgb_intensities[:, 0].tolist()
    results_dict['Mean Green Relative Intensity'] = rgb_intensities[:, 1].tolist()
    results_dict['Mean Blue Relative Intensity'] = rgb_intensities[:, 2].tolist()

    return results_dict


default_image_properties = [
    {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'numerical'},
    {'name': 'Area', 'method': area, 'output_type': 'numerical'},
    {'name': 'Brightness', 'method': brightness, 'output_type': 'numerical'},
    {'name': 'RMS Contrast', 'method': rms_contrast, 'output_type': 'numerical'},
    {'name': 'Mean Red Relative Intensity', 'method': mean_red_relative_intensity, 'output_type': 'numerical'},
    {'name': 'Mean Green Relative Intensity', 'method': mean_green_relative_intensity, 'output_type': 'numerical'},
    {'name': 'Mean Blue Relative Intensity', 'method': mean_blue_relative_intensity, 'output_type': 'numerical'}
]
