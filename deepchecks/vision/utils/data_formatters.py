from typing import Callable, Optional

import numpy as np
from torch.utils.data import DataLoader


class DataFormatter:
    """
    Class for formatting the image data outputted from the dataloader to the required format for check displays.

    Parameters
    ----------
    data_formatter : Callable
        Function that takes in a batch of data and returns the data in the following format (a batch of cv2 images):
        A 4D numpy array, with the first dimension being the batch size, the second being the image height (y axis), the
        third being the image width (x axis), and the fourth being the number of channels. The numbers in the array
        should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while grayscale
        images should have 1 channel.
    """

    def __init__(self, data_formatter: Callable):
        self.data_formatter = data_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.data_formatter(*args, **kwargs)

    def validate_data(self, batch_data) -> Optional[str]:
        """Validate that the data is in the required format.

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
        print(batch_data.shape)

        if not isinstance(batch_data, np.ndarray):
            return 'The data must be a numpy array.'
        if batch_data.ndim != 4:
            return 'The data must be a 4D array.'
        if batch_data.shape[3] not in [1, 3]:
            return 'The data must have 1 or 3 channels.'
        if batch_data.min() < 0 or batch_data.max() > 255:
            return 'The data must be in the range [0, 255].'
