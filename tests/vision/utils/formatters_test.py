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
import numpy as np

from deepchecks.vision import VisionDataset
from deepchecks.vision.utils import ClassificationLabelFormatter
from hamcrest import assert_that, instance_of, calling, raises, equal_to

from deepchecks.vision.utils.data_formatters import DataFormatter
from tests.vision.vision_conftest import *


def test_classification_formatter_invalid_dataloader(three_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(three_tuples_dataloader)
    assert_that(err, equal_to("Check requires dataloader to return tuples of (input, label)."))


def test_classification_formatter_formatting_valid_label_shape(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to(None))


def test_classification_formatter_formatting_invalid_label_type(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: [x, x])

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to("Check requires classification label to be a torch.Tensor or numpy array"))


def numpy_shape_dataloader(shape, value: float = 1):
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return np.ones(shape) * value

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4, collate_fn=np.stack)


def test_data_formatter_missing_dimensions():
    formatter = DataFormatter(lambda x: x)

    err = formatter.validate_data(numpy_shape_dataloader((10, 10)))
    assert_that(err, equal_to('The data must be a 4D array.'))


def test_data_formatter_wrong_color_channel():
    formatter = DataFormatter(lambda x: x)

    err = formatter.validate_data(numpy_shape_dataloader((3, 10, 10)))
    assert_that(err, equal_to('The data must have 1 or 3 channels.'))


def test_data_formatter_invalid_values():
    formatter = DataFormatter(lambda x: x * 300)

    err = formatter.validate_data(numpy_shape_dataloader((10, 10, 3)))
    assert_that(err, equal_to('The data must be in the range [0, 255].'))

    formatter = DataFormatter(lambda x: -x)

    err = formatter.validate_data(numpy_shape_dataloader((10, 10, 3)))
    assert_that(err, equal_to('The data must be in the range [0, 255].'))


def test_data_formatter_valid_dimensions():
    formatter = DataFormatter(lambda x: x)

    err = formatter.validate_data(numpy_shape_dataloader((10, 10, 3)))
    assert_that(err, equal_to(None))