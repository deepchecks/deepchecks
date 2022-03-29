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

import numpy as np

from deepchecks.vision import Batch, VisionData
from deepchecks.vision.checks.distribution.abstract_property_outliers import AbstractPropertyOutliers
from deepchecks.vision.utils import image_properties

__all__ = ['ImagePropertyOutliers']


class ImagePropertyOutliers(AbstractPropertyOutliers):

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
        return image_properties.default_image_properties
