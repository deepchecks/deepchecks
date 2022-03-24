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
"""Module contains the similar image leakage check."""
from collections import defaultdict
from typing import Any, Callable, TypeVar, Hashable, Dict, Union
import numpy as np
import pandas as pd
from PIL.Image import fromarray
from imagehash import average_hash

from deepchecks import ConditionResult
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_utils.single_feature_contribution_utils import get_single_feature_contribution, \
    get_single_feature_contribution_per_class
from deepchecks.utils.strings import format_number
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.vision.utils import image_properties
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.vision_data import TaskType

__all__ = ['SimilarImageLeakage']


class SimilarImageLeakage(TrainTestCheck):
    #TODO Complete doc
    """

    Parameters
    ----------
    n_top_show: int, default: 5
        Number of images to show, sorted by the similarity score between them
    hash_size: int, default: 8
        Size of hashed image. Algorithm will hash the image to a hash_size*hash_size binary image. #TODO: Improve
    similarity_threshold: float, default: 0.01
        range (0,1) #TODO: Improve
    """

    def __init__(
            self,
            n_top_show: int = 10,
            hash_size: int = 8,
            similarity_threshold: float = 0.01
    ):
        super().__init__()
        self.n_top_show = n_top_show
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.min_pixel_diff = int(np.ceil(similarity_threshold * hash_size**2))

    def initialize_run(self, context: Context):
        self._hashed_train_images = []
        self._hashed_test_images = []

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches.""" #TODO
        hashed_images = [average_hash(fromarray(img), hash_size=self.hash_size) for img in batch.images]

        if dataset_kind == DatasetKind.TRAIN:
            self._hashed_train_images += hashed_images
        else:
            self._hashed_test_images += hashed_images

    def compute(self, context: Context) -> CheckResult:
        """Calculate the PPS between each property and the label. #TODO

        Returns
        -------
        CheckResult
            value: dictionaries of PPS values for train, test and train-test difference.
            display: bar graph of the PPS of each feature.
        """

        test_hashes = np.array(self._hashed_test_images)

        similar_images = []

        for i, h in enumerate(self._hashed_train_images):
            is_similar = (test_hashes - h) < self.min_pixel_diff
            if any(is_similar):
                similar_images.append(i)

        ret_value = similar_images
        display = []

        return CheckResult(value=ret_value, display=display, header='Similar Image Leakage')
