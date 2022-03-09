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
"""Test functions of the VISION train test label drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling, empty, instance_of

import numpy as np
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData
from deepchecks.vision.checks import SimpleFeatureContributionTrainTest
from deepchecks.vision.utils import ImageFormatter, DetectionLabelFormatter, ClassificationLabelFormatter
from tests.vision.vision_conftest import *

from deepchecks.vision.utils.transformations import un_normalize_batch


def mnist_image_formatter(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    return un_normalize_batch(tensor, (0.1307,), (0.3081,))


# Create bias in the image_formatter that will
def mnist_image_formatter_with_bias(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
    for i, label in enumerate(batch[1]):
        ret[i] = ret[i].clip(min=5 * label, max=180 + 5 * label)
    return ret

def test_no_drift_classification(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = SimpleFeatureContributionTrainTest()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': instance_of(dict),
        'test': instance_of(dict),
        'train-test difference': instance_of(dict)
    })
                )


def test_drift_classification(mnist_data_loader_train, mnist_data_loader_test):
    # Arrange
    train = VisionData(mnist_data_loader_train, image_formatter=ImageFormatter(mnist_image_formatter),
                       label_formatter=ClassificationLabelFormatter(lambda arr: [int(x) for x in arr[1]]))
    test = VisionData(mnist_data_loader_test, image_formatter=ImageFormatter(mnist_image_formatter_with_bias),
                      label_formatter=ClassificationLabelFormatter(lambda arr: [int(x) for x in arr[1]]))

    check = SimpleFeatureContributionTrainTest()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': instance_of(dict),
        'test': has_entries({'brightness': close_to(0.43, 0.01)}),
        'train-test difference': instance_of(dict)
    })
                )
