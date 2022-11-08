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
"""Module of weak segments performance check."""
from typing import Dict, Callable, List, Any, Union, Optional
from deepchecks.vision import Context, SingleDatasetCheck
from ignite.metrics import Metric

from deepchecks.vision.utils.image_properties import default_image_properties


class WeakSegmentsPerformance(SingleDatasetCheck):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.
    """

    def __init__(
        self,
        scorers: Union[Dict[str, Union[Metric, Callable, str]], List[Any]] = None,
        image_properties: List[Dict[str, Any]] = None,
        number_of_bins: int = 5,
        number_of_samples_to_infer_bins: int = 1000,
        n_to_show: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_properties = image_properties if image_properties else default_image_properties
        self.scorers = scorers
        self.number_of_bins = number_of_bins
        self.number_of_samples_to_infer_bins = number_of_samples_to_infer_bins
        self.n_to_show = n_to_show

        self._state = None
        
