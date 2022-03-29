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
from typing import TypeVar, List, Tuple
import numpy as np
from PIL.Image import fromarray
from imagehash import average_hash

from deepchecks import ConditionResult, ConditionCategory
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Context, TrainTestCheck, Batch

__all__ = ['SimilarImageLeakage']

from deepchecks.vision.utils.image_functions import prepare_thumbnail

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

    _THUMBNAIL_SIZE = (200, 200)

    def __init__(
            self,
            n_top_show: int = 10,
            hash_size: int = 8,
            similarity_threshold: float = 0.05
    ):
        super().__init__()
        if not (isinstance(n_top_show, int) and (n_top_show >= 0)):
            raise DeepchecksValueError('n_top_show must be a positive integer')
        self.n_top_show = n_top_show
        if not (isinstance(hash_size, int) and (hash_size >= 0)):
            raise DeepchecksValueError('hash_size must be a positive integer')
        self.hash_size = hash_size
        if not (isinstance(similarity_threshold, float) and (similarity_threshold >= 0) and
                (similarity_threshold <= 1)):
            raise DeepchecksValueError('similarity_threshold must be a float in range (0,1)')
        self.similarity_threshold = similarity_threshold
        self.min_pixel_diff = int(np.ceil(similarity_threshold * (hash_size**2 / 2)))

    def initialize_run(self, context: Context):
        self._hashed_train_images = []
        self._hashed_test_images = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
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

        train_hashes = np.array(self._hashed_train_images)

        similar_indices = {
            'train': [],
            'test': []
        }

        for i, h in enumerate(self._hashed_test_images):
            is_similar = (train_hashes - h) < self.min_pixel_diff
            if any(is_similar):
                for j in np.argwhere(is_similar):  # Return indices where True
                    similar_indices['train'].append(j[0])  # append only the first similar image in train
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
        similar_pairs = None
        if similar_indices['test']:
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

            html = HTML_TEMPLATE.format(
                count=len(similar_indices['test']),
                n_of_images=len(display_indices),
                train_images=''.join(display_images['train']),
                test_images=''.join(display_images['test']),
            )

            display.append(html)

            # return tuples of indices in original respective dataset objects
            similar_pairs = list(zip(
                context.train.to_dataset_index(*similar_indices['train']),
                context.test.to_dataset_index(*similar_indices['test'])
            ))

        return CheckResult(value=similar_pairs, display=display, header='Similar Image Leakage')

    def add_condition_similar_images_not_more_than(self: SIL, threshold: int = 0) -> SIL:
        """Add new condition.

        Add condition that will check the number of similar images is not greater than X.
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
            num_similar_images = len(set([t[1] for t in value]))

            if num_similar_images > threshold:
                message = f'Number of similar images between train and test datasets: {num_similar_images}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Number of similar images between train and test is not greater than '
                                  f'{threshold}', condition)


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
        grid-template-rows: auto 1fr 1fr;
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

