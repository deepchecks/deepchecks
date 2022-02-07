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
from hamcrest import assert_that, equal_to, calling, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_formatters import ImageFormatter
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


def numpy_shape_dataloader(shape, value: float = 1, collate_fn=None):

    if collate_fn is None:
        collate_fn = np.stack

    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return np.ones(shape) * value

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4, collate_fn=collate_fn)


def test_data_formatter_not_iterable():
    formatter = ImageFormatter()

    batch = 1
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, 'The batch data must be an iterable.')
    )


def test_data_formatter_not_numpy():
    formatter = ImageFormatter(lambda x: [[x] for x in x])

    batch = next(iter(numpy_shape_dataloader((10, 10, 3))))
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, 'The data inside the iterable must be a numpy array.')
    )


def test_data_formatter_missing_dimensions():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 10))))
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, 'The data inside the iterable must be a 3D array.')
    )


def test_data_formatter_wrong_color_channel():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((3, 10, 10))))
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, 'The data inside the iterable must have 1 or 3 channels.')
    )


def test_data_formatter_invalid_values():
    formatter = ImageFormatter(lambda x: x * 300)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3))))
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, r'The data inside the iterable must be in the range \[0, 255\].')
    )

    formatter = ImageFormatter(lambda x: -x)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3))))
    assert_that(
        calling(formatter.validate_data).with_args(batch),
        raises(DeepchecksValueError, r'The data inside the iterable must be in the range \[0, 255\].')
    )


def test_data_formatter_valid_dimensions():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3))))
    err = formatter.validate_data(batch)
    assert_that(err, equal_to(None))


def test_data_formatter_valid_dimensions_other_iterable():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3), collate_fn=list)))
    err = formatter.validate_data(batch)
    assert_that(err, equal_to(None))

    batch = next(iter(numpy_shape_dataloader((10, 10, 3), collate_fn=tuple)))
    err = formatter.validate_data(batch)
    assert_that(err, equal_to(None))
