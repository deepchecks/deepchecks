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
from typing import Union

import numpy as np
from hamcrest import assert_that, equal_to, calling, raises, close_to

from deepchecks.vision.utils import ClassificationLabelFormatter
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_formatters import ImageFormatter
# pylint: disable=wildcard-import,redefined-outer-name,unused-wildcard-import
from tests.vision.vision_conftest import *


def test_classification_formatter_invalid_dataloader(three_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(three_tuples_dataloader)
    assert_that(err, equal_to('Check requires dataloader to return tuples of (input, label).'))


def test_classification_formatter_formatting_valid_label_shape(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to(None))


def test_classification_formatter_formatting_invalid_label_type(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: [x, x])

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to('Check requires classification label to be a torch.Tensor or numpy array'))


def numpy_shape_dataloader(shape: tuple = None, value: Union[float, np.array] = 1, collate_fn=None):

    if collate_fn is None:
        collate_fn = np.stack

    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            if isinstance(value, (float, int)):
                return np.ones(shape) * value
            else:
                return value

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
    formatter.validate_data(batch)


def test_data_formatter_valid_dimensions_other_iterable():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3), collate_fn=list)))
    formatter.validate_data(batch)

    batch = next(iter(numpy_shape_dataloader((10, 10, 3), collate_fn=tuple)))
    formatter.validate_data(batch)


def test_brightness_grayscale():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.zeros((3, 10, 1)),  np.ones((7, 10, 1))], axis=0)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.brightness(batch)

    assert_that(res, equal_to([0.7]*4))


def test_brightness_rgb():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.ones((10, 10, 1)) * 1,
                            np.ones((10, 10, 1)) * 2,
                            np.ones((10, 10, 1)) * 3], axis=2)

    expected_result = 0.299 + 0.587 * 2 + 0.114 * 3

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.brightness(batch)

    assert_that(res, equal_to([expected_result]*4))


def test_aspect_ratio():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 20, 3))))

    res = formatter.aspect_ratio(batch)

    assert_that(res, equal_to([0.5]*4))


def test_area():
    formatter = ImageFormatter(lambda x: x)

    batch = next(iter(numpy_shape_dataloader((10, 20, 3))))

    res = formatter.area(batch)

    assert_that(res, equal_to([200]*4))


def test_normalized_mean_red():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.ones((10, 10, 1)) * 1,
                            np.ones((10, 10, 1)) * 2,
                            np.ones((10, 10, 1)) * 3], axis=2)

    expected_result = 1/6

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.normalized_red_mean(batch)

    assert_that(res[0], close_to(expected_result, 0.0000001))


def test_normalized_mean_green():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.ones((10, 10, 1)) * 1,
                            np.ones((10, 10, 1)) * 2,
                            np.ones((10, 10, 1)) * 3], axis=2)

    expected_result = 2/6

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.normalized_green_mean(batch)

    assert_that(res[0], close_to(expected_result, 0.0000001))


def test_normalized_mean_blue():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.ones((10, 10, 1)) * 1,
                            np.ones((10, 10, 1)) * 2,
                            np.ones((10, 10, 1)) * 3], axis=2)

    expected_result = 3/6

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.normalized_blue_mean(batch)

    assert_that(res[0], close_to(expected_result, 0.0000001))


def test_flatten_batch_without_sample():
    formatter = ImageFormatter(lambda x: x, sample_size_for_image_properties=None)

    value = np.concatenate([np.ones((2, 2, 1)) * 1,
                            np.ones((2, 2, 1)) * 2,
                            np.ones((2, 2, 1)) * 3], axis=2)

    expected_result = np.concatenate([np.ones((4, 1)) * 1,
                                      np.ones((4, 1)) * 2,
                                      np.ones((4, 1)) * 3], axis=1)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter._flatten_batch(batch)  # pylint: disable=protected-access

    assert_that(np.array_equal(res[0], expected_result), equal_to(True))


def test_flatten_batch_with_sampling():
    formatter = ImageFormatter(lambda x: x, sample_size_for_image_properties=3)

    value = np.concatenate([np.ones((2, 2, 1)) * 1,
                            np.ones((2, 2, 1)) * 2,
                            np.ones((2, 2, 1)) * 3], axis=2)

    expected_result = np.concatenate([np.ones((3, 1)) * 1,
                                      np.ones((3, 1)) * 2,
                                      np.ones((3, 1)) * 3], axis=1)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter._flatten_batch(batch)  # pylint: disable=protected-access

    assert_that(np.array_equal(res[0], expected_result), equal_to(True))


def test_flatten_batch_with_sampling_larger_than_num_pixels():
    formatter = ImageFormatter(lambda x: x, sample_size_for_image_properties=5)

    value = np.concatenate([np.ones((2, 2, 1)) * 1,
                            np.ones((2, 2, 1)) * 2,
                            np.ones((2, 2, 1)) * 3], axis=2)

    expected_result = np.concatenate([np.ones((4, 1)) * 1,
                                      np.ones((4, 1)) * 2,
                                      np.ones((4, 1)) * 3], axis=1)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter._flatten_batch(batch)  # pylint: disable=protected-access

    assert_that(np.array_equal(res[0], expected_result), equal_to(True))
