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
"""Tests for weak segment performance check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_items, has_length, calling, raises, any_of, all_of, has_property, has_properties, has_entries
from sklearn.metrics import f1_score, make_scorer

from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks import ConditionCategory
from deepchecks.tabular.checks.model_evaluation import PerformanceBias
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
            check = PerformanceBias(protected_feature=feat1)
            check.run(test, model)

            for feat2 in control_feat_to_test:
                check = PerformanceBias(protected_feature=feat1, control_feature=feat2)
                check.run(test, model)

    # Act
    for task in tasks:
        run_task(*task)

    # Assert
    pass # no error


def test_run_value_error(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceBias(protected_feature="sex")
    check_invalid1 = PerformanceBias(protected_feature="invalid_feature")
    check_invalid2 = PerformanceBias(protected_feature="sex", control_feature="invalid_feature")
    check_invalid3 = PerformanceBias(protected_feature="sex", control_feature="sex")

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
        raises(DeepchecksValueError, r'protected_feature and control_feature cannot be the same')
    )


def test_condition_fail(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceBias("sex")
    check.add_condition_bounded_performance_difference(lower_bound=-0.03)
    check2 = PerformanceBias("sex")
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
    check = PerformanceBias("sex")
    check.add_condition_bounded_performance_difference(lower_bound=-0.04)
    check2 = PerformanceBias("sex")
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
    check = PerformanceBias("sex")

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
    assert_that(result.display, has_length(2))
    assert_that(result.value['scores_df'].round(3).to_dict(), has_entries(expected_value.round(3).to_dict()))


def test_numeric_values_classwise(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceBias("sex", scorer="f1_per_class")

    expected_value = pd.DataFrame({
        'sex': {1: ' Female', 0: ' Male'},
        '_scorer': {1: 'f1_per_class', 0: 'f1_per_class'},
        '_score': {1: 0.9557802895102122, 0: 0.6198331788693234},
        '_class': {1: ' <=50K', 0: ' >50K'},
        '_baseline': {1: 0.9054560599750104, 0: 0.5940497480084539},
        '_baseline_count': {1: 16281, 0: 16281},
        '_count': {1: 5421, 0: 10860},
        '_diff': {1: 0.05032422953520177, 0: 0.02578343086086954}
    })

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result.display, has_length(2))
    assert_that(result.value['scores_df'].round(3).to_dict(), has_entries(expected_value.round(3).to_dict()))


def test_numeric_values_classwise_with_control(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceBias("sex", control_feature="race", scorer="f1_per_class")

    expected_value = pd.DataFrame({
        'sex': {5: ' Female',
                3: ' Female',
                4: ' Female',
                0: ' Male',
                2: ' Male',
                1: ' Male'},
        'race': {5: 'Others',
                3: ' White',
                4: ' Black',
                0: ' White',
                2: 'Others',
                1: ' Black'},
        '_scorer': {5: 'f1_per_class',
                3: 'f1_per_class',
                4: 'f1_per_class',
                0: 'f1_per_class',
                2: 'f1_per_class',
                1: 'f1_per_class'},
        '_score': {5: 0.9447619047619047,
                3: 0.9524044389642415,
                4: 0.9786354238456237,
                0: 0.621011989433042,
                2: 0.6484375,
                1: 0.5596330275229359},
        '_class': {5: ' <=50K',
                3: ' <=50K',
                4: ' <=50K',
                0: ' >50K',
                2: ' >50K',
                1: ' >50K'},
        '_baseline': {5: 0.9048760991207034,
                3: 0.8991080632871679,
                4: 0.9554229554229555,
                0: 0.5966672639311952,
                2: 0.5993265993265993,
                1: 0.5347985347985348},
        '_baseline_count': {5: 774, 3: 13946, 4: 1561, 0: 13946, 2: 774, 1: 1561},
        '_count': {5: 283, 3: 4385, 4: 753, 0: 9561, 2: 491, 1: 808},
        '_diff': {5: 0.039885805641201255,
                3: 0.05329637567707368,
                4: 0.02321246842266822,
                0: 0.024344725501846853,
                2: 0.04911090067340074,
                1: 0.02483449272440108}
    })

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result.display, has_length(2))
    assert_that(result.value['scores_df'].round(3).to_dict(), has_entries(expected_value.round(3).to_dict()))


def test_na_scores_on_small_subgroups(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check = PerformanceBias("sex", min_subgroup_size=1_000_000_000)

    # Act
    result = check.run(test, model)

    # Assert
    assert_that(result.value['scores_df']["_score"].isna().all())
    assert_that(result.value['scores_df']["_baseline"].isna().all())


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
        check = PerformanceBias("sex", scorer=scorer)
        check.run(test, model)

    # Assert
    pass # no error
