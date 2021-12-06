# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import typing as t
import re
from hamcrest import assert_that, instance_of, only_contains, matches_regexp

from deepchecks import Dataset, CheckResult, ConditionCategory
from deepchecks.checks.methodology import ModelInferenceTimeCheck

from tests.checks.utils import equal_condition_result, SCIENTIFIC_NOTATION_REGEXP


def test_model_inference_time_check(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck()

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result, instance_of(CheckResult))
    assert_that(result.value, instance_of(float))
    assert_that(result.display, instance_of(list))
    assert_that(result.display, only_contains(instance_of(str))) # type: ignore

    details_pattern = (
        r'Average model inference time for one sample \(in seconds\): '
        fr'{SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )

    assert_that(result.display[0], matches_regexp(details_pattern)) # type: ignore


def test_model_inference_time_check_with_condition_that_should_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than(0.1)

    # Act
    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    name = (
        'Average model inference time for one sample is not '
        'greater than 0.1'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        category=ConditionCategory.FAIL,
        name=name,
        details=''
    ))


def test_model_inference_time_check_with_condition_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than(0.00000001)

    # Act
    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    name = (
        'Average model inference time for one sample is not '
        'greater than 1e-08'
    )
    details_pattern = re.compile(
        r'Average model inference time for one sample \(in seconds\) '
        fr'is {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        category=ConditionCategory.FAIL,
        name=name,
        details=details_pattern
    ))
