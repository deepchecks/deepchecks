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
from hamcrest import assert_that, instance_of, calling, raises, is_
from unittest.mock import patch

from deepchecks import vision
from deepchecks.vision.datasets.detection.coco import (
    load_dataset,
    DATA_DIR,
    CocoDataset
)
from torchvision.datasets.utils import download_and_extract_archive


def patch_side_effect(*args, **kwargs):
    print("hi i'm being called")
    return download_and_extract_archive(*args, **kwargs)


@patch('torchvision.datasets.utils.download_and_extract_archive', )
def test_load_dataset(mock_download_and_extract_archive):
    # mock object should call original function
    mock_download_and_extract_archive.side_effect = patch_side_effect
    print(mock_download_and_extract_archive)

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
        assert_that(mock_download_and_extract_archive.called, is_(False))
        verify(loader)
        assert_that(loader, instance_of(DataLoader))
    print('hisdfoasdhfoadsfh')
    print(mock_download_and_extract_archive.called)
    print('fdsasdfasd')
    assert_that(7, 8)


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
