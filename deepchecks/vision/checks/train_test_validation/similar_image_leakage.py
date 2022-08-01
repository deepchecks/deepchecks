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
import random
from typing import List, Tuple, TypeVar

import numpy as np
from imagehash import average_hash
from PIL.Image import fromarray

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Batch, Context, TrainTestCheck

__all__ = ['SimilarImageLeakage']

from deepchecks.vision.utils.image_functions import prepare_thumbnail

SIL = TypeVar('SIL', bound='SimilarImageLeakage')


class SimilarImageLeakage(TrainTestCheck):
    """Check for images in training that are similar to images in test.

    Parameters
    ----------
    n_top_show: int, default: 5
        Number of images to show, sorted by the similarity score between them
    hash_size: int, default: 8
        Size of hashed image. Algorithm will hash the image to a hash_size*hash_size binary image. Increasing this
        value will increase the accuracy of the algorithm, but will also increase the time and memory requirements.
    similarity_threshold: float, default: 0.1
        Similarity threshold (0,1). The similarity score defines what is the ratio of pixels that are different between
        the two images. If the similarity score is below the threshold, the images are considered similar.
        Note: The threshold is defined such that setting it to 1 will result in similarity being detected for all images
        with up to half their pixels differing from each other. For a value of 1, random images (which on average
        differ from each other by half their pixels) will be detected as similar half the time. To further illustrate,
        for a hash of 8X8, setting the score to 1 will result with all images with up to 32 different pixels being
        considered similar.
    """

    _THUMBNAIL_SIZE = (200, 200)

    def __init__(
            self,
            n_top_show: int = 10,
            hash_size: int = 8,
            similarity_threshold: float = 0.1,
            **kwargs):
        super().__init__(**kwargs)
        if not (isinstance(n_top_show, int) and (n_top_show >= 0)):
            raise DeepchecksValueError('n_top_show must be a positive integer')
        self.n_top_show = n_top_show
        if not (isinstance(hash_size, int) and (hash_size >= 0)):
            raise DeepchecksValueError('hash_size must be a positive integer')
        self.hash_size = hash_size
        if not (isinstance(similarity_threshold, (float, int)) and (0 <= similarity_threshold <= 1)):
            raise DeepchecksValueError('similarity_threshold must be a float in range (0,1)')
        self.similarity_threshold = similarity_threshold

        self.min_pixel_diff = int(np.ceil(similarity_threshold * (hash_size**2 / 2)))

    def initialize_run(self, context: Context):
        """Initialize the run by initializing the lists of image hashes."""
        self._hashed_train_images = []
        self._hashed_test_images = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate image hashes for train and test."""
        hashed_images = [average_hash(fromarray(img.squeeze()), hash_size=self.hash_size) for img in batch.images]

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
        train_hashes = np.array(self._hashed_train_images)

        similar_indices = {
            'train': [],
            'test': []
        }

        for i, h in enumerate(self._hashed_test_images):
            is_similar = (train_hashes - h) < self.min_pixel_diff
            if any(is_similar):
                # For each test image with similarities, append the first similar image in train
                similar_indices['train'].append(np.argwhere(is_similar)[0][0])
                similar_indices['test'].append(i)

        display_indices = random.sample(range(len(similar_indices['test'])),
                                        min(self.n_top_show, len(similar_indices['test'])))

        display_images = {
            'train': [],
            'test': []
        }

        data_obj = {
            'train': context.train,
            'test': context.test
        }

        display = []
        similar_pairs = []
        if similar_indices['test']:

            # TODO: this for loop should be below `if context.with_display:` branch
            for similar_index in display_indices:
                for dataset in ('train', 'test'):
                    image = data_obj[dataset].batch_to_images(
                        data_obj[dataset].batch_of_index(similar_indices[dataset][similar_index])
                    )[0]
                    image_thumbnail = prepare_thumbnail(
                        image=image,
                        size=self._THUMBNAIL_SIZE,
                        copy_image=False
                    )
                    display_images[dataset].append(image_thumbnail)

            # return tuples of indices in original respective dataset objects
            similar_pairs = list(zip(
                context.train.to_dataset_index(*similar_indices['train']),
                context.test.to_dataset_index(*similar_indices['test'])
            ))

            if context.with_display:
                html = HTML_TEMPLATE.format(
                    count=len(similar_indices['test']),
                    n_of_images=len(display_indices),
                    train_images=''.join(display_images['train']),
                    test_images=''.join(display_images['test']),
                )

                display.append(html)

        return CheckResult(value=similar_pairs, display=display, header='Similar Image Leakage')

    def add_condition_similar_images_less_or_equal(self: SIL, threshold: int = 0) -> SIL:
        """Add condition - number of similar images is less or equal to the threshold.

        The condition count how many unique images in test are similar to those in train.

        Parameters
        ----------
        threshold : int , default: 0
            Number of allowed unique images in test that are similar to train

        Returns
        -------
        SIL
        """

        def condition(value: List[Tuple[int, int]]) -> ConditionResult:
            num_similar_images = len(set(t[1] for t in value))
            message = f'Number of similar images between train and test datasets: {num_similar_images}' \
                if num_similar_images > 0 else 'Found 0 similar images between train and test datasets'
            category = ConditionCategory.PASS if num_similar_images <= threshold else ConditionCategory.FAIL
            return ConditionResult(category, message)

        return self.add_condition(f'Number of similar images between train and test is less or equal to {threshold}',
                                  condition)


HTML_TEMPLATE = """
<h3><b>Similar Images</b></h3>
<div>
Total number of test samples with similar images in train: {count}
</div>
<h4>Samples</h4>
<div
    style="
        overflow-x: auto;
        display: grid;
        grid-template-columns: auto repeat({n_of_images}, 1fr);
        grid-gap: 1.5rem;
        justify-items: center;
        align-items: center;
        padding: 2rem;
        width: max-content;">
    <h5>Train</h5>{train_images}
    <h5>Test</h5>{test_images}
</div>
"""
