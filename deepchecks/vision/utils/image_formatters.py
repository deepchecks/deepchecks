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
from typing import Callable, Optional, Tuple, List

import numpy as np

from deepchecks.core.errors import DeepchecksValueError

__all__ = ['ImageFormatter']


class ImageFormatter:
    """Class for formatting the image data outputted from the dataloader to the required format for check displays.

    Parameters
    ----------
    image_formatter : Callable, optional
        Function that takes in a batch from DataLoader and returns only the data from it in the following format (an
        iterable of cv2 images):
        Each image in the iterable must be a [H, W, C] 3D numpy array. The first dimension must be the image height
        (y axis), the second being the image width (x axis), and the third being the number of channels. The numbers
        in the array should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while
        grayscale images should have 1 channel. The batch itself can be any iterable - a list of arrays, a tuple of
        arrays or simply a 4D numpy array, when the first dimension is the batch dimension.
        If None, the identity function will be used (the data will be assumed to be in format as-is).
    sample_size_for_image_properties: int, optional. default: 1000
        sampling size used for image property estimation.
        If defined, samples only N pixels (uniformly) out of image and calculates property on those.
        If None, calculates on the whole image.
    """

    def __init__(self, image_formatter: Optional[Callable] = None,
                 sample_size_for_image_properties: Optional[int] = 1000):
        if image_formatter is None:
            self.data_formatter = lambda x: x[0]
        else:
            self.data_formatter = image_formatter

        self.sample_size_for_image_properties = sample_size_for_image_properties

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.data_formatter(*args, **kwargs)

    def validate_data(self, batch):
        """Validate that the data is in the required format.

        The validation is done on the first element of the batch.

        Parameters
        ----------
        batch

        Raises
        -------
        DeepchecksValueError
            If the batch data doesn't fit the format after being transformed by self().

        """
        data = self(batch)
        try:
            sample: np.ndarray = data[0]
        except TypeError as err:
            raise DeepchecksValueError('The batch data must be an iterable.') from err
        if not isinstance(sample, np.ndarray):
            raise DeepchecksValueError('The data inside the iterable must be a numpy array.')
        if sample.ndim != 3:
            raise DeepchecksValueError('The data inside the iterable must be a 3D array.')
        if sample.shape[2] not in [1, 3]:
            raise DeepchecksValueError('The data inside the iterable must have 1 or 3 channels.')
        if sample.min() < 0 or sample.max() > 255:
            raise DeepchecksValueError('The data inside the iterable must be in the range [0, 255].')
        if np.all(sample <= 1):
            raise DeepchecksValueError('The data inside the iterable appear to be normalized.')

    def aspect_ratio(self, batch: List[np.array]) -> List[float]:
        """Return list of floats of image height to width ratio."""
        return [x[0] / x[1] for x in self._sizes(batch)]

    def area(self, batch: List[np.array]) -> List[int]:
        """Return list of integers of image areas (height multiplied by width)."""
        return [np.prod(self.get_size(img)) for img in batch]

    def brightness(self, batch: List[np.array]) -> List[float]:
        """Calculate brightness on each image in the batch."""
        if self._is_grayscale(batch) is True:
            return [img.mean() for img in batch]
        else:
            flattened_batch = self._flatten_batch(batch)
            return [(0.299*img[:, 0] + 0.587*img[:, 1] + 0.114 * img[:, 2]).mean() for img in flattened_batch]

    def contrast(self,  batch: List[np.array]) -> List[float]:
        """Return constrast of image."""
        raise NotImplementedError('Not yet implemented')  # TODO

    def normalized_red_mean(self, batch: List[np.array]) -> List[float]:
        """Return the normalized mean of the red channel."""
        return [x[0] for x in self._normalized_rgb_mean(batch)]

    def normalized_green_mean(self, batch: List[np.array]) -> List[float]:
        """Return the normalized mean of the green channel."""
        return [x[1] for x in self._normalized_rgb_mean(batch)]

    def normalized_blue_mean(self, batch: List[np.array]) -> List[float]:
        """Return the normalized mean of the blue channel."""
        return [x[2] for x in self._normalized_rgb_mean(batch)]

    def _sizes(self, batch: List[np.array]):
        """Return list of tuples of image height and width."""
        return [self.get_size(img) for img in batch]

    def _normalized_rgb_mean(self, batch: List[np.array]) -> List[Tuple[float, float, float]]:
        """Calculate normalized mean for each channel (rgb) in image.

        The normalized mean of each channel is calculated by first normalizing the image's pixels (meaning, each color
        is normalized to its relevant intensity, by dividing the color intensity by the other colors). Then, the mean
        for each image channel is calculated.

        Parameters
        ----------
        batch: List[np.array]
            A list of arrays, each arrays represents an image in the required deepchecks format.

        Returns
        -------
        List[np.array]:
            List of 3-dimensional arrays, each dimension is the normalized mean of the color channel. An array is
            returned for each image.
        """
        if self._is_grayscale(batch) is True:
            return [(None, None, None)] * len(batch)

        flattened_batch = self._flatten_batch(batch)
        # TODO: Check for faster implementations than pixel by pixel
        normalized_images = [np.array([self._normalize_colors_in_pixel(pxl) for pxl in img]) for img in flattened_batch]
        return [img.mean(axis=0) for img in normalized_images]

    @staticmethod
    def _normalize_colors_in_pixel(pixel):
        pxl_sum = pixel.sum()
        return np.array([pixel[i] / pxl_sum if pxl_sum else 0 for i in range(3)])

    def _is_grayscale(self, batch):
        return self.get_dimension(batch[0]) == 1

    def _sample_images_in_batch(self, flattened_batch):
        if self.sample_size_for_image_properties is None:
            raise RuntimeError(
                'function sample_images_in_batch should not be called if sample_size_for_image_properties is None')
        ret_batch = []
        np.random.seed(len(flattened_batch))
        for img in flattened_batch:
            if img.shape[0] <= self.sample_size_for_image_properties:
                ret_batch.append(img)
            else:
                indexes = np.random.randint(0, img.shape[0], self.sample_size_for_image_properties)
                sampled_img = np.array([img[i, :] for i in indexes])
                ret_batch.append(sampled_img)

        return ret_batch

    def _flatten_batch(self, batch):
        if self._is_grayscale(batch) is True:
            raise DeepchecksValueError('function _flatten_batch cannot run on 1-dimensional image (grayscale)')
        flattened_imgs = [img.reshape([img.shape[0] * img.shape[1], 3]) for img in batch]
        if self.sample_size_for_image_properties is not None:
            return self._sample_images_in_batch(flattened_imgs)
        else:
            return flattened_imgs

    @staticmethod
    def get_size(img) -> Tuple[int, int]:
        """Get size of image as (height, width) tuple."""
        return img.shape[0], img.shape[1]

    @staticmethod
    def get_dimension(img) -> int:
        """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
        return img.shape[2]
