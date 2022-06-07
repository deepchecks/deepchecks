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
"""Test functions of the train test drift."""
from hamcrest import assert_that, close_to, equal_to, has_entries

from deepchecks.tabular.checks import TrainTestFeatureDrift
from tests.base.utils import equal_condition_result


def test_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, model)
    print(result.value)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.69, 0.01)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.34, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.31, 0.01)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
    }))


def test_drift_no_model(drifted_data_and_model):
    # Arrange
    train, test, _ = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.34, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
    }))


def test_drift_max_drift_score_condition_fail(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.2 and numerical drift score < 0.1',
        details='Failed for 2 out of 4 columns.\n'
                'Found 1 categorical columns with PSI above threshold: {\'categorical_with_drift\': \'0.22\'}\n'
                'Found 1 numeric columns with Earth Mover\'s Distance above threshold: '
                '{\'numeric_with_drift\': \'0.34\'}'
    ))


def test_drift_max_drift_score_condition_fail_cramer(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='cramer_v').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.2 and numerical drift score < 0.1',
        details='Failed for 2 out of 4 columns.\n'
                'Found 1 categorical columns with Cramer\'s V above threshold: {\'categorical_with_drift\': \'0.23\'}\n'
                'Found 1 numeric columns with Earth Mover\'s Distance above threshold: '
                '{\'numeric_with_drift\': \'0.34\'}'
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=1,
                                             max_allowed_numeric_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Passed for 4 columns out of 4 columns.\n'
                'Found column "categorical_with_drift" has the highest categorical drift score: 0.22\n'
                'Found column "numeric_with_drift" has the highest numerical drift score: 0.34',
        name='categorical drift score < 1 and numerical drift score < 1'
    ))


def test_drift_max_drift_score_multi_columns_drift_pass(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_less_than(allowed_num_features_exceeding_threshold=2)
    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Passed for 2 columns out of 4 columns.\n'
                'Found column "categorical_with_drift" has the highest categorical drift score: 0.22\n'
                'Found column "numeric_with_drift" has the highest numerical drift score: 0.34',
        name='categorical drift score < 0.2 and numerical drift score < 0.1'
    ))