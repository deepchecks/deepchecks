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
"""Test functions of the train test drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to

from deepchecks.checks import TrainTestFeatureDrift
from tests.checks.utils import equal_condition_result


def test_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift()

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.31, 0.01)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.25, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.69, 0.01)}
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
    check = TrainTestFeatureDrift()

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
            {'Drift score': close_to(0.25, 0.01),
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
    check = TrainTestFeatureDrift().add_condition_drift_score_not_greater_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='PSI and Earth Mover\'s Distance cannot be greater than 0.2 and 0.1 respectively',
        details='Found categorical columns with PSI over 0.2: categorical_with_drift\n'
                'Found numeric columns with Earth Mover\'s Distance over 0.1: numeric_with_drift'
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestFeatureDrift().add_condition_drift_score_not_greater_than(max_allowed_psi_score=1,
                                                                               max_allowed_earth_movers_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='PSI and Earth Mover\'s Distance cannot be greater than 1 and 1 respectively'
    ))
