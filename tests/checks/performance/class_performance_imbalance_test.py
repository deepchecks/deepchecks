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
import typing as t
import re
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from hamcrest.core.matcher import Matcher
from hamcrest import (
    assert_that, instance_of, has_property, all_of,
    equal_to, has_entries, only_contains, calling, raises
)

from deepchecks import Dataset, CheckResult, ConditionCategory
from deepchecks.checks import ClassPerformanceImbalance
from deepchecks.errors import DeepchecksValueError

from tests.checks.utils import equal_condition_result


def test_class_performance_imbalance(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalance()
    # Act
    check_result = check.run(dataset=test, model=model)
    # Assert
    validate_class_performance_imbalance_check_result(check_result)


def test_init_class_performance_imbalance_with_empty_dict_of_scorers():
    assert_that(
        calling(ClassPerformanceImbalance).with_args(alternative_scorers=dict()),
        raises(DeepchecksValueError, 'Scorers dictionary can\'t be empty')
    )


def test_init_class_performance_imbalance_with_scorers_dict_that_contains_not_callable_and_not_name_of_sklearn_scorer():
    assert_that(
        calling(ClassPerformanceImbalance).with_args(alternative_scorers=dict(Metric=1)),
        raises(
            DeepchecksValueError,
            r"Scorer Metric value should be either a callable or string but got: int"
        )
    )


def test_class_performance_imbalance_with_custom_scorers(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    alternative_scorers = {
        'Test1': lambda model, features, labels: np.array([1,2,3]),
        'Test2': lambda model, features, labels: np.array([1,2,3]),
        'Test3': lambda model, features, labels: np.array([1,2,3])
    }
    check = ClassPerformanceImbalance(alternative_scorers=alternative_scorers)

    # Act
    check_result = check.run(dataset=test, model=model)

    # Assert
    class_scores_matcher = has_entries({
        k: instance_of(int)
        for k in alternative_scorers.keys()
    })
    value_matcher = all_of(
        instance_of(dict),
        has_entries({
            0: all_of(instance_of(dict), class_scores_matcher),
            1: all_of(instance_of(dict), class_scores_matcher),
            2: all_of(instance_of(dict), class_scores_matcher),
        })
    )

    validate_class_performance_imbalance_check_result(
        check_result,
        value=value_matcher
    )


def test_class_performance_imbalance_with_custom_scorers_that_return_not_array(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model

    alternative_scorers = {'Test1': lambda model, features, labels: 1,}
    check = ClassPerformanceImbalance(alternative_scorers=alternative_scorers)

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r"Expected scorer Test1 to return np.ndarray but got: int"
        )
    )


def test_class_performance_imbalance_with_custom_scorers_that_return_empty_array(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrage
    _, test, model = iris_split_dataset_and_model

    # dict dtype is allowed but it cannot be empty
    alternative_scorers = {'Test3': lambda model, features, labels: np.array([])}
    check = ClassPerformanceImbalance(alternative_scorers=alternative_scorers)

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r"Expected scorer to return array of length 3, but got length 0"
        )
    )


def test_class_performance_imbalance_with_custom_scorers_that_return_array_with_incorrect_dtype(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model

    # dict dtype is allowed but values dtype must be int|float
    alternative_scorers = {'Test3': lambda model, features, labels: np.array(["a", "b", "c"])}
    check = ClassPerformanceImbalance(alternative_scorers=alternative_scorers)

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r"Expected scorer Test3 to return np.ndarray of number kind but got: U"
        )
    )


def test_condition_ratio_difference_not_greater_than_threshold_that_should_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalance().add_condition_ratio_difference_not_greater_than(0.5)

    # Act
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)

    condition_result, *_ = check.conditions_decision(check_result)

    # Assert
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        name=(
            "Relative ratio difference between labels 'F1' score "
            "is not greater than 50.00%"
        ),
        details='',
        category=ConditionCategory.FAIL,
    ))


def test_condition_ratio_difference_not_greater_than_threshold_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalance().add_condition_ratio_difference_not_greater_than(0.12)

    # Act
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)
    condition_result, *_ = check.conditions_decision(check_result)

    # Assert
    detail_pattern = re.compile(
        r'Relative ratio difference between highest and lowest classes is greater than 12\.00%\. '
        r'Score: F1, lowest class: \d+, highest class: \d+;'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        name=(
            "Relative ratio difference between labels 'F1' score "
            "is not greater than 12.00%"
        ),
        details=detail_pattern,
        category=ConditionCategory.FAIL,
    ))


def test_add_condition_for_unknown_score_function():
    assert_that(
        calling(ClassPerformanceImbalance().add_condition_ratio_difference_not_greater_than).with_args(0.12, 'acuracy'),
        raises(DeepchecksValueError, 'Unknown score function  - acuracy')
    )


def validate_class_performance_imbalance_check_result(
    result: CheckResult,
    value: t.Optional[Matcher] = None
):
    if value is None:
        default_scores_matcher = has_entries({
            'F1': instance_of(float),
            'Precision': instance_of(float),
            'Recall': instance_of(float),
        })

        value_matcher = all_of(
            instance_of(dict),
            has_entries({
                0: all_of(instance_of(dict), default_scores_matcher),
                1: all_of(instance_of(dict), default_scores_matcher),
                2: all_of(instance_of(dict), default_scores_matcher),
            })
        )
    else:
        value_matcher = value

    assert_that(result, all_of(
        instance_of(CheckResult),
        has_property('value', value_matcher),
        has_property('header', all_of(instance_of(str), equal_to('Class Performance Imbalance'))),
        has_property('display', only_contains(has_property('__call__')))
    ))

