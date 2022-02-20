# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import time
from torch.utils.data import DataLoader
from hamcrest import assert_that, instance_of, calling, raises

from deepchecks import vision
from deepchecks.vision.datasets.detection.coco import (
    load_dataset,
    DATA_DIR,
    CocoDataset
)


def test_load_dataset():
    def verify(loader):
        assert_that(loader, instance_of(DataLoader))
        assert_that(loader.dataset, instance_of(CocoDataset))
        assert_that(loader.dataset.train is True)
        assert_that((DATA_DIR / 'coco128' / 'images').exists())
        assert_that((DATA_DIR / 'coco128' / 'labels').exists())

    if not (DATA_DIR / 'coco128').exists():
        loader = load_dataset(train=True, object_type='DataLoader')
        verify(loader)
        test_load_dataset()
    else:
        # verifying that downloaded prev data was used and not re-downloaded
        start = time.time()
        loader = load_dataset(train=True, object_type='DataLoader')
        end = time.time()
        verify(loader)
        assert_that(loader, instance_of(DataLoader))
        assert_that((end - start) < 2, "Downloaded previously data was not used!")


def test_deepchecks_dataset_load():
    loader = load_dataset(train=True, object_type='VisionData')
    assert_that(loader, instance_of(vision.VisionData))


def test__load_dataset__func_with_unknow_object_type_parameter():
    assert_that(
        calling(load_dataset).with_args(object_type="<unknonw>"),
        raises(TypeError)
    )


def test_train_test_split():
    train = load_dataset(train=True, object_type='DataLoader')
    test = load_dataset(train=False, object_type='DataLoader')
    
    assert_that((len(train.dataset) + len(test.dataset)) == 128)

    train_images = set(it.name for it in train.dataset.images)
    test_images = set(it.name for it in test.dataset.images)

    intersection = train_images.intersection(test_images)
    assert_that(len(intersection) == 0)
