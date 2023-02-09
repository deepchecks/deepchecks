# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import numpy as np

from hamcrest import assert_that, instance_of, equal_to, is_not

from deepchecks.vision import VisionData
from deepchecks.vision.datasets.classification.mnist_tensorflow import load_dataset, load_model


def test_deepchecks_dataset_load():
    dataset = load_dataset(train=True)
    assert_that(dataset, instance_of(VisionData))

    dataset = load_dataset(train=False)
    assert_that(dataset, instance_of(VisionData))


def test_regular_visiondata_with_shuffle():
    np.random.seed(42)
    vision_data = load_dataset(n_samples=100, shuffle=False)
    batch = next(iter(vision_data))
    vision_data_again = load_dataset(n_samples=100, shuffle=False)
    batch_again = next(iter(vision_data_again))
    vision_data_shuffled = load_dataset(n_samples=100, shuffle=True)
    batch_shuffled = next(iter(vision_data_shuffled))
    vision_data_shuffled_again = load_dataset(n_samples=100, shuffle=True)
    batch_shuffled_again = next(iter(vision_data_shuffled_again))

    assert_that(batch['labels'][0], is_not(equal_to(batch_shuffled['labels'][0])))
    assert_that(batch['labels'][0], equal_to(batch_again['labels'][0]))
    assert_that(batch_shuffled_again['labels'][0], is_not(equal_to(batch_shuffled['labels'][0])))

    assert_that(batch['predictions'][0][0], is_not(equal_to(batch_shuffled['predictions'][0][0])))
    assert_that(batch['predictions'][0][0], equal_to(batch_again['predictions'][0][0]))
    assert_that(batch_shuffled_again['predictions'][0][0], is_not(equal_to(batch_shuffled['predictions'][0][0])))
