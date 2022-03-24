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
from typing import Any, TypeVar, List, Tuple
import numpy as np
from PIL.Image import fromarray
from imagehash import average_hash

from deepchecks import ConditionResult, ConditionCategory
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import Context, TrainTestCheck

__all__ = ['SimilarImageLeakage']

SIL = TypeVar('SIL', bound='SimilarImageLeakage')


class SimilarImageLeakage(TrainTestCheck):
    """

    Parameters
    ----------
    n_top_show: int, default: 5
        Number of images to show, sorted by the similarity score between them
    hash_size: int, default: 8
        Size of hashed image. Algorithm will hash the image to a hash_size*hash_size binary image. #TODO: Improve
    similarity_threshold: float, default: 0.1
        range (0,1) #TODO: Improve
    """

    def __init__(
            self,
            n_top_show: int = 10,
            hash_size: int = 8,
            similarity_threshold: float = 0.05
    ):
        super().__init__()
        self.n_top_show = n_top_show
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.min_pixel_diff = int(np.ceil(similarity_threshold * (hash_size**2 / 2)))

    def initialize_run(self, context: Context):
        self._hashed_train_images = []
        self._hashed_test_images = []

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Calculate image hashes for train and test."""
        hashed_images = [average_hash(fromarray(img), hash_size=self.hash_size) for img in batch.images]

        if dataset_kind == DatasetKind.TRAIN:
            self._hashed_train_images += hashed_images
        else:
            self._hashed_test_images += hashed_images

    def compute(self, context: Context) -> CheckResult:
        """Find similar images by comparing image hashes between train and test.

        Returns
        -------
        CheckResult
            value: list of tuples of similar image instances, in format (train_index, test_index). The index is by the
                order of the images deepchecks received the images.
            display: pairs of similar images
        """

        test_hashes = np.array(self._hashed_test_images)

        similar_images = []

        for i, h in enumerate(self._hashed_train_images):
            is_similar = (test_hashes - h) < self.min_pixel_diff
            if any(is_similar):
                for j in np.argwhere(is_similar):  # Return indices where True
                    similar_images.append((i, j[0]))

        ret_value = similar_images
        display = []

        return CheckResult(value=ret_value, display=display, header='Similar Image Leakage')

    def add_condition_similar_images_not_more_than(self: SIL, threshold: int = 10) -> SIL:
        """Add new condition.

        Add condition that will check the number of similar images is not greater than X.
        The condition count how many unique images in test are similar to those in train.

        Parameters
        ----------
        threshold : int , default: 10
            number of unique images in test that are similar to train

        Returns
        -------
        SIL
        """

        def condition(value: List[Tuple[int, int]]) -> ConditionResult:
            num_similar_images = len(set([t[0] for t in value]))

            if num_similar_images:
                message = f'Number of similar images between train and test datasets: {num_similar_images}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Number of similar images between train and test is not greater than '
                                  f'{threshold}', condition)

