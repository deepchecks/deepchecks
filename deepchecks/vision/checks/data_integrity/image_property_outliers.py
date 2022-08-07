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
"""Module of ImagePropertyOutliers check."""
import typing as t

import numpy as np

from deepchecks.vision import Batch, VisionData
from deepchecks.vision.checks.data_integrity.abstract_property_outliers import AbstractPropertyOutliers
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType

__all__ = ['ImagePropertyOutliers']


class ImagePropertyOutliers(AbstractPropertyOutliers):
    """Find outliers images with respect to the given properties.

    The check computes several image properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`property guide </user-guide/vision/vision_properties.rst>`
    n_show_top : int , default: 5
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    """

    def __init__(self,
                 image_properties: t.List[t.Dict[str, t.Any]] = None,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 1.5,
                 **kwargs):
        super().__init__(properties_list=image_properties, property_input_type=PropertiesInputType.IMAGES,
                         n_show_top=n_show_top, iqr_percentiles=iqr_percentiles,
                         iqr_scale=iqr_scale, **kwargs)

    def get_relevant_data(self, batch: Batch):
        """Get the data on which the check calculates outliers for."""
        return batch.images

    def draw_image(self, data: VisionData, sample_index: int, index_of_value_in_sample: int,
                   num_properties_in_sample: int) -> np.ndarray:
        """Return an image to show as output of the display.

        Parameters
        ----------
        data : VisionData
            The vision data object used in the check.
        sample_index : int
            The batch index of the sample to draw the image for.
        index_of_value_in_sample : int
            Each sample property is list, then this is the index of the outlier in the sample property list.
        num_properties_in_sample
            The number of values in the sample's property list.
        """
        return data.batch_to_images(data.batch_of_index(sample_index))[0]

    def get_default_properties(self, data: VisionData):
        """Return default properties to run in the check."""
        return default_image_properties
