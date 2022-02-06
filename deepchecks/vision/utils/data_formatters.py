from typing import Callable, Optional

import numpy as np


class DataFormatter:
    """
    Class for formatting the image data outputted from the dataloader to the required format for check displays.

    Parameters
    ----------
    data_formatter : Callable
        Function that takes in a batch of data and returns the data in the following format (an iterable of cv2 images):
        Each image in the iterable must be a [H, W, C] 3D numpy array. The first dimension must be the image height
        (y axis), the second being the image width (x axis), and the third being the number of channels. The numbers
        in the array should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while
        grayscale images should have 1 channel. The batch itself can be any iterable - a list of arrays, a tuple of
        arrays or simply a 4D numpy array, when the first dimension is the batch dimension.
    """

    def __init__(self, data_formatter: Callable):
        self.data_formatter = data_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.data_formatter(*args, **kwargs)

    def validate_data(self, batch_data) -> Optional[str]:
        """Validate that the data is in the required format.

        The validation is done on the first element of the batch.

        Parameters
        ----------
        batch_data
            A batch of data outputted from the dataloader.

        Returns
        -------
        Optional[str]
            None if the data is valid, otherwise a string containing the error message.

        """

        batch_data = self(batch_data)
        try:
            sample: np.ndarray = batch_data[0]
        except TypeError:
            return 'The batch data must be an iterable.'
        if sample.ndim != 3:
            return 'The data inside the iterable must be a 3D array.'
        if sample.shape[2] not in [1, 3]:
            return 'The data must have 1 or 3 channels.'
        if sample.min() < 0 or sample.max() > 255:
            return 'The data must be in the range [0, 255].'
