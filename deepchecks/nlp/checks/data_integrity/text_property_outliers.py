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
# TODO: Prototype, go over and make sure code+docs+tests are good
"""Module of TextPropertyOutliers check."""
import typing as t

import numpy as np

from deepchecks.nlp import TextData
from deepchecks.nlp.utils.text_properties import default_text_properties
from deepchecks.nlp.checks.data_integrity.abstract_property_outliers import AbstractPropertyOutliers
from deepchecks.nlp.utils.nlp_properties import PropertiesInputType

__all__ = ['TextPropertyOutliers']

from deepchecks.nlp.utils.text_utils import trim


class TextPropertyOutliers(AbstractPropertyOutliers):
    """Find outliers images with respect to the given properties.

    The check computes several image properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    text_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    n_show_top : int , default: 5
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    """

    def __init__(self,
                 text_properties: t.List[t.Dict[str, t.Any]] = None,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 1.5,
                 **kwargs):
        super().__init__(properties_list=text_properties, property_input_type=PropertiesInputType.TEXT,
                         n_show_top=n_show_top, iqr_percentiles=iqr_percentiles,
                         iqr_scale=iqr_scale, **kwargs)

    def plot_text(self, data: TextData, sample_index: int, index_of_value_in_sample: int,
                   num_properties_in_sample: int) -> np.ndarray:
        """Return an image to show as output of the display.

        Parameters
        ----------
        data : TextData
            The text data object used in the check.
        sample_index : int
            The batch index of the sample to draw the image for.
        index_of_value_in_sample : int
            Each sample property is list, then this is the index of the outlier in the sample property list.
        num_properties_in_sample
            The number of values in the sample's property list.
        """
        return trim(data.text[sample_index], 100)

    def get_default_properties(self, data: TextData):
        """Return default properties to run in the check."""
        return default_text_properties
