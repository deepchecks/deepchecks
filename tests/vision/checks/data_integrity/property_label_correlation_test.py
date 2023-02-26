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
import numpy as np
from hamcrest import assert_that, close_to, contains_exactly, has_entries

from deepchecks.vision.checks import PropertyLabelCorrelation
from tests.base.utils import equal_condition_result


def med_prop(batch):
    return [np.median(x) for x in batch]


def mean_prop(batch):
    return [np.mean(x) for x in batch]


def test_classification_without_bias(mnist_visiondata_train):
    result = PropertyLabelCorrelation().add_condition_property_pps_less_than().run(mnist_visiondata_train)
    # assert check result
    assert_that(result.value, has_entries({'Brightness': close_to(0.079, 0.005), 'Area': close_to(0.0, 0.005)}))
    # assert condition
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        details='Passed for all of the properties',
        name='Properties\' Predictive Power Score is less than 0.8'))


def test_classification_with_bias(mnist_train_brightness_bias):
    result = PropertyLabelCorrelation().add_condition_property_pps_less_than(0.2).run(mnist_train_brightness_bias)
    # assert check result
    assert_that(result.value, has_entries({'Brightness': close_to(0.248, 0.005), 'Area': close_to(0.0, 0.005)}))
    # assert condition
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details='Found 1 out of 7 properties with PPS above threshold: {\'Brightness\': \'0.25\'}',
        name='Properties\' Predictive Power Score is less than 0.2'))


def test_classification_with_alternative_properties(mnist_visiondata_train):
    alt_props = [{'name': 'med', 'method': med_prop, 'output_type': 'numerical'},
                 {'name': 'mean', 'method': mean_prop, 'output_type': 'numerical'}]
    result = PropertyLabelCorrelation(image_properties=alt_props).run(mnist_visiondata_train)
    assert_that(result.value.keys(), contains_exactly('mean', 'med'))
    assert_that(result.value['med'], close_to(0.0, 0.005))


def test_object_detection_without_bias(coco_visiondata_train):
    result = PropertyLabelCorrelation().run(coco_visiondata_train)
    assert_that(result.value, has_entries({'Brightness': close_to(0, 0.005), 'Area': close_to(0.016557, 0.005)}))


def test_object_detection_with_bias(coco_train_brightness_bias):
    result = PropertyLabelCorrelation().run(coco_train_brightness_bias)
    assert_that(result.value, has_entries({'Brightness': close_to(0.0876, 0.01), 'Area': close_to(0.0187, 0.01)}))
