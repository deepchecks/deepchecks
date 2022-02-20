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
from skimage.color import rgb2gray
from torch.utils.data import DataLoader, Dataset

from deepchecks.vision.utils import ClassificationLabelFormatter
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_formatters import ImageFormatter
# pylint: disable=wildcard-import,redefined-outer-name,unused-wildcard-import
from tests.vision.vision_conftest import *


def numpy_shape_dataloader(shape: tuple = None, value: Union[float, np.array] = 255, collate_fn=None):
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


def test_classification_formatter_formatting_valid_label_shape(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter()

    # Assert no exception raised
    formatter.validate_label(next(iter(two_tuples_dataloader)))


def test_classification_formatter_formatting_invalid_label_type(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: [x, x])

    # Assert
    assert_that(calling(formatter.validate_label).with_args(next(iter(two_tuples_dataloader))),
                raises(DeepchecksValueError,
                       'Check requires classification label to be a torch.Tensor or numpy array'))


def test_data_formatter_not_iterable():
    formatter = ImageFormatter(lambda x: x)

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
    formatter = ImageFormatter()

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

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.brightness(batch)

    assert_that(res[0], close_to(1.86, 0.01))


def test_blur_grayscale():
    formatter = ImageFormatter(lambda x: x)

    np.random.seed(42)
    value = np.random.randint(0, 255, (100, 100, 1))

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.blur(batch)

    assert_that(res[0], close_to(0.12, 0.01))


def test_blur_rgb():
    formatter = ImageFormatter(lambda x: x)

    np.random.seed(42)
    value = np.random.randint(0, 255, (100, 100, 3))

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.blur(batch)

    assert_that(res[0], close_to(0.12, 0.01))


def test_rms_contrast_grayscale():
    formatter = ImageFormatter(lambda x: x)

    value = np.concatenate([np.zeros((3, 10, 1)),  np.ones((7, 10, 1))], axis=0)

    expected_value = np.sqrt((70 * 0.3**2 + 30 * 0.7**2) / 100)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.rms_contrast(batch)

    assert_that(res[0], equal_to(expected_value))


def test_rms_contrast_rgb():
    formatter = ImageFormatter(lambda x: x)

    # Create image that after turning from rgb to grayscale is 30% value 0 and 70% value 3:
    value = np.concatenate([np.zeros((3, 10, 3)),  np.concatenate([np.ones((7, 10, 1)) * 1/0.2125,
                                                                   np.ones((7, 10, 1)) * 1/0.7154,
                                                                   np.ones((7, 10, 1)) * 1/0.0721], axis=2)],
                           axis=0)

    expected_value = np.sqrt((210 * 0.9**2 + 90 * 2.1**2) / 300)

    batch = next(iter(numpy_shape_dataloader(value=value)))

    res = formatter.rms_contrast(batch)

    assert_that(res[0], close_to(expected_value, 0.00001))


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
