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
"""Tests for weak segment performance check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_items, has_length, calling, raises, any_of, all_of, has_property, has_properties, has_entries
from sklearn.metrics import f1_score, make_scorer

from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks import ConditionCategory
from deepchecks.tabular.checks.model_evaluation import PerformanceDisparityReport
from tests.base.utils import equal_condition_result


def test_no_error(adult_split_dataset_and_model, avocado_split_dataset_and_model):
    # Arrange
    tasks = [
        (
            *adult_split_dataset_and_model, 
            ["sex", "age"], 
            ["education", "capital-gain"]
        ),
        (
            *avocado_split_dataset_and_model, 
            ["type", "year"], 
            ["region", "Total Bags"]
        ),
    ]
    def run_task(train, test, model, protected_feat_to_test, control_feat_to_test):
        train = train.sample()
        test = test.sample()

        for feat1 in protected_feat_to_test:
            check = PerformanceDisparityReport(protected_feature=feat1)
            check.run(test, model)

            for feat2 in control_feat_to_test:
                check = PerformanceDisparityReport(protected_feature=feat1, control_feature=feat2)
                check.run(test, model)

    # Act
    for task in tasks:
        run_task(*task)

    # Assert
    pass # no error


def test_run_value_error(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceDisparityReport(protected_feature="sex")
    check_invalid1 = PerformanceDisparityReport(protected_feature="invalid_feature")
    check_invalid2 = PerformanceDisparityReport(protected_feature="sex", control_feature="invalid_feature")
    check_invalid3 = PerformanceDisparityReport(protected_feature="sex", control_feature="sex")

    # Act & Assert
    assert_that(
        calling(check.run).with_args("invalid_data"), 
        raises(DeepchecksValueError, r'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )
    assert_that(
        calling(check.run).with_args(test), 
        raises(DeepchecksNotSupportedError, r'Check is irrelevant for Datasets without model')
    )
    assert_that(
        calling(check_invalid1.run).with_args(test, model), 
        raises(DeepchecksValueError, r'Feature invalid_feature not found in dataset.')
    )
    assert_that(
        calling(check_invalid2.run).with_args(test, model), 
        raises(DeepchecksValueError, r'Feature invalid_feature not found in dataset.')
    )
    assert_that(
        calling(check_invalid3.run).with_args(test, model), 
        raises(DeepchecksValueError, r'protected_feature sex and control_feature sex are the same.')
    )


def test_condition_fail(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceDisparityReport("sex")
    check.add_condition_bounded_performance_difference(lower_bound=-0.03)
    check2 = PerformanceDisparityReport("sex")
    check2.add_condition_bounded_relative_performance_difference(lower_bound=-0.04)

    # Act
    result = check.run(test, model)
    condition_result = result.conditions_results
    result2 = check2.run(test, model)
    condition_result2 = result2.conditions_results

    # Assert
    assert_that(condition_result, has_items(has_properties(
        category=ConditionCategory.FAIL,
        name="Performance differences are bounded between -0.03 and inf.",
        details="Found 1 subgroups with performance differences outside of the given bounds."
    )))
    assert_that(condition_result2, has_items(has_properties(
        category=ConditionCategory.FAIL,
        name="Relative performance differences are bounded between -0.04 and inf.",
        details="Found 1 subgroups with relative performance differences outside of the given bounds."
    )))


def test_condition_pass(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceDisparityReport("sex")
    check.add_condition_bounded_performance_difference(lower_bound=-0.04)
    check2 = PerformanceDisparityReport("sex")
    check2.add_condition_bounded_relative_performance_difference(lower_bound=-0.042)

    # Act
    result = check.run(test, model)
    condition_result = result.conditions_results
    result2 = check2.run(test, model)
    condition_result2 = result2.conditions_results

    # Assert
    assert_that(condition_result, has_items(has_properties(
        category=ConditionCategory.PASS,
        name="Performance differences are bounded between -0.04 and inf.",
        details="Found 0 subgroups with performance differences outside of the given bounds."
    )))
    assert_that(condition_result2, has_items(has_properties(
        category=ConditionCategory.PASS,
        name="Relative performance differences are bounded between -0.042 and inf.",
        details="Found 0 subgroups with relative performance differences outside of the given bounds."
    )))


def test_numeric_values(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceDisparityReport("sex")

    expected_value = pd.DataFrame({
        'sex': {0: ' Male', 1: ' Female'},
        '_scorer': {0: 'Accuracy', 1: 'Accuracy'},
        '_score': {0: 0.8111418047882136, 1: 0.9177273565762775},
        '_baseline': {0: 0.8466310423192679, 1: 0.8466310423192679},
        '_baseline_count': {0: 16281, 1: 16281},
        '_count': {0: 10860, 1: 5421},
        '_diff': {0: -0.03548923753105426, 1: 0.07109631425700957}
    })

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result.display, has_length(1))
    assert_that(result.display[0].data, has_length(2))
    assert_that(result.value.round(3).to_dict(), has_entries(expected_value.round(3).to_dict()))


def test_na_scores_on_small_subgroups(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceDisparityReport("sex", min_subgroup_size=1_000_000_000)

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result.value["_score"].isna().all())
    assert_that(result.value["_baseline"].isna().all())


def test_scorers_types_no_error(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    scorer_types = [
        None,
        "f1",
        ('f1_score', make_scorer(f1_score, average='micro')),
        {'f1_score': make_scorer(f1_score, average='micro')},
    ]
    
    # Act
    for scorer in scorer_types:
        check = PerformanceDisparityReport("sex", scorer=scorer)
        check.run(test, model)

    # Assert
    pass # no error
