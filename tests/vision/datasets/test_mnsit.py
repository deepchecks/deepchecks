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
import time
from torch.utils.data import DataLoader
from hamcrest import assert_that, instance_of, calling, raises

from deepchecks.vision import VisionData
from deepchecks.vision.datasets.classification.mnist import (
    load_dataset,
    load_model,
    MNistNet,
    DATA_PATH,
    MODEL_PATH
)


def test_dataset_load():
    dataloader = load_dataset(object_type="DataLoader")
    assert_that(dataloader, instance_of(DataLoader))
    assert_that(DATA_PATH.exists() and DATA_PATH.is_dir())
    assert_that(dataloader.dataset._check_exists() is True)


def test_deepchecks_dataset_load():
    dataloader, dataset = load_dataset(object_type='DataLoader'), load_dataset(object_type='VisionData')
    assert_that(dataset, instance_of(VisionData))
    assert_that(dataloader, instance_of(DataLoader))


def test__load_dataset__func_with_unknow_object_type_parameter():
    assert_that(
        calling(load_dataset).with_args(object_type="<unknonw>"),
        raises(TypeError)
    )


def test_pretrained_model_load():
    if MODEL_PATH.exists():
        start = time.time()
        model = load_model()
        end = time.time()
        assert_that((end - start) < 1, "Saved model was not used!")
        assert_that(model.training is False)
        assert_that(model, instance_of(MNistNet))
    else:
        model = load_model()
        assert_that(model.training is False)
        assert_that(model, instance_of(MNistNet))
        assert_that(MODEL_PATH.exists() and MODEL_PATH.is_file())
        # to verify loading from the file
        test_pretrained_model_load()