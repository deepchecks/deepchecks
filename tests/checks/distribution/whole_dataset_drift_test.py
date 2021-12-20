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
"""Test functions of the whole dataset drift check."""
from hamcrest import assert_that, has_entries, close_to

from deepchecks import Dataset
from deepchecks.checks import WholeDatasetDrift
from tests.checks.utils import equal_condition_result


def test_no_drift(drifted_data):

    # Arrange
    train_ds, test_ds = drifted_data
    train_ds = Dataset(train_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']),
                       label_name=train_ds.label_name)
    test_ds = Dataset(test_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']),
                      label_name=test_ds.label_name)
    check = WholeDatasetDrift()

    # Act
    result = check.run(train_ds, test_ds)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.5, 0.03),
        'domain_classifier_feature_importance': has_entries(
            {'categorical_without_drift': close_to(0.81, 0.001),
             'numeric_without_drift': close_to(0.2, 0.02)}
        ),
    }))


def test_drift(drifted_data):

    # Arrange
    train_ds, test_ds = drifted_data
    check = WholeDatasetDrift()

    # Act
    result = check.run(train_ds, test_ds)

    # Assert
    assert_that(result.value, has_entries({
        'domain_classifier_auc': close_to(0.9, 0.031),
        'domain_classifier_feature_importance': has_entries(
            {'categorical_without_drift': close_to(0, 0.02),
             'numeric_without_drift': close_to(0, 0.02),
             'categorical_with_drift': close_to(0, 0.02),
             'numeric_with_drift': close_to(1, 0.02)
             }
        ),
    }))


def test_max_drift_score_condition_pass(drifted_data):
    # Arrange
    train_ds, test_ds = drifted_data
    train_ds = Dataset(train_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']),
                       label_name=train_ds.label_name)
    test_ds = Dataset(test_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']),
                      label_name=test_ds.label_name)
    check = WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than()

    # Act
    result = check.run(train_ds, test_ds)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='Drift value is not greater than 0.25',
    ))


def test_max_drift_score_condition_fail(drifted_data):
    # Arrange
    train_ds, test_ds = drifted_data
    check = WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than()

    # Act
    result = check.run(train_ds, test_ds)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Drift value is not greater than 0.25',
        details='Found drift value of: 0.86, corresponding to a domain classifier AUC of: 0.93'
    ))
