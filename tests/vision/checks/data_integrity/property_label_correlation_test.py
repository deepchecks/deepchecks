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
"""Test for the check property label correlation."""
from copy import copy

import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, has_entries, contains_exactly

from deepchecks.vision.checks import PropertyLabelCorrelation
from deepchecks.vision.utils.transformations import un_normalize_batch
from tests.base.utils import equal_condition_result
from tests.vision.vision_conftest import *


def mnist_batch_to_images_with_bias(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
    for i, label in enumerate(batch[1]):
        label = label.cpu().detach()
        ret[i] = ret[i].clip(min=5 * label, max=180 + 5 * label)
    return ret


def med_prop(batch):
    return [np.median(x) for x in batch]


def mean_prop(batch):
    return [np.mean(x) for x in batch]


def test_classification_without_bias(mnist_dataset_train, device):
    result = PropertyLabelCorrelation().run(mnist_dataset_train, device=device)
    assert_that(result.value, has_entries({'Brightness': close_to(0.0737, 0.005), 'Area': close_to(0.0, 0.005)}))


def test_classification_with_bias(mnist_dataset_train, device):
    mnist_dataset_train.batch_to_images = mnist_batch_to_images_with_bias
    result = PropertyLabelCorrelation().run(mnist_dataset_train, device=device)
    assert_that(result.value, has_entries({'Brightness': close_to(0.234, 0.005), 'Area': close_to(0.0, 0.005)}))


def test_classification_with_alternative_properties(mnist_dataset_train, device):
    alt_props = [{'name': 'med', 'method': med_prop, 'output_type': 'numerical'},
                 {'name': 'mean', 'method': mean_prop, 'output_type': 'numerical'}]
    result = PropertyLabelCorrelation(image_properties=alt_props).run(mnist_dataset_train, device=device)
    assert_that(result.value.keys(), contains_exactly('med', 'mean'))
    assert_that(result.value['med'], close_to(1.0, 0.005))
