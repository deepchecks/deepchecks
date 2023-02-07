# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import re
import typing as t

from hamcrest import assert_that, instance_of, matches_regexp, only_contains

from deepchecks.core import CheckResult, ConditionCategory
from deepchecks.tabular.checks.model_evaluation import ModelInferenceTime
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import SCIENTIFIC_NOTATION_REGEXP, equal_condition_result


def test_model_inference_time_check(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTime()

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
    check = ModelInferenceTime().add_condition_inference_time_less_than(0.1)

    # Act
    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    name = 'Average model inference time for one sample is less than 0.1'
    details_pattern = re.compile(
        fr'Found average inference time \(seconds\): {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        category=ConditionCategory.PASS,
        name=name,
        details=details_pattern
    ))


def test_model_inference_time_check_with_condition_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTime().add_condition_inference_time_less_than(0.00000001)

    # Act
    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    name = 'Average model inference time for one sample is less than 1e-08'
    details_pattern = re.compile(
        fr'Found average inference time \(seconds\): {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        category=ConditionCategory.FAIL,
        name=name,
        details=details_pattern
    ))
