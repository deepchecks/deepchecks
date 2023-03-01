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
from unittest.mock import patch

from hamcrest import assert_that, calling, equal_to, instance_of, is_, raises
from torch.utils.data import DataLoader

from deepchecks.vision import VisionData
from deepchecks.vision.datasets.detection.mask import MASK_DIR, MaskDataset, load_dataset


def patch_side_effect(*args, **kwargs):
    return MaskDataset.download_mask(*args, **kwargs)


def load_dataset_test(mock_download_and_extract_archive):
    def verify(loader):
        assert_that(loader, instance_of(DataLoader))
        assert_that(loader.dataset, instance_of(MaskDataset))
        assert_that(loader.dataset.train is True)
        assert_that((MASK_DIR / 'mask' / 'images').exists())
        assert_that((MASK_DIR / 'mask' / 'annotations').exists())

    if not (MASK_DIR / 'mask').exists():
        loader = load_dataset(day_index=0, object_type='DataLoader')
        verify(loader)
        mock_download_and_extract_archive.reset_mock()
        load_dataset_test(mock_download_and_extract_archive)
    else:
        # verifying that downloaded prev data was used and not re-downloaded
        loader = load_dataset(day_index=0, object_type='DataLoader')
        assert_that(mock_download_and_extract_archive.called, is_(False))
        verify(loader)
        assert_that(loader, instance_of(DataLoader))


def test_deepchecks_dataset_load():
    loader = load_dataset(day_index=0, object_type='VisionData')
    assert_that(loader, instance_of(VisionData))


def test__load_dataset__func_with_unknow_object_type_parameter():
    assert_that(
        calling(load_dataset).with_args(object_type="<unknonw>"),
        raises(TypeError)
    )


def test_train_test_split():
    train = load_dataset(day_index=0, object_type='DataLoader')
    test = load_dataset(day_index=30, object_type='DataLoader')

    assert_that((len(train.dataset) + len(test.dataset)), equal_to(1706))
