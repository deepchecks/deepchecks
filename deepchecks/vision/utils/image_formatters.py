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
from typing import Callable, Optional

import numpy as np

from deepchecks.core.errors import DeepchecksValueError


__all__ = ['ImageFormatter']


class ImageFormatter:
    """Class for formatting the image data outputted from the dataloader to the required format for check displays.

    Parameters
    ----------
    image_formatter : Callable, optional
        Function that takes in a batch of data and returns the data in the following format (an iterable of cv2 images):
        Each image in the iterable must be a [H, W, C] 3D numpy array. The first dimension must be the image height
        (y axis), the second being the image width (x axis), and the third being the number of channels. The numbers
        in the array should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while
        grayscale images should have 1 channel. The batch itself can be any iterable - a list of arrays, a tuple of
        arrays or simply a 4D numpy array, when the first dimension is the batch dimension.
        If None, the identity function will be used (the data will be assumed to be in format as-is).
    """

    def __init__(self, image_formatter: Optional[Callable] = None):
        if image_formatter is None:
            self.data_formatter = lambda x: x
        else:
            self.data_formatter = image_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.data_formatter(*args, **kwargs)

    def validate_data(self, batch_data):
        """Validate that the data is in the required format.

        The validation is done on the first element of the batch.

        Parameters
        ----------
        batch_data
            A batch of data outputted from the dataloader.

        Raises
        -------
        DeepchecksValueError
            If the batch data doesn't fit the format after being transformed by self().

        """
        batch_data = self(batch_data)
        try:
            sample: np.ndarray = batch_data[0]
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
