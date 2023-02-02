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
"""Tests for Mixed Nulls check"""
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_entries, has_items, has_length, instance_of

from deepchecks.tabular.checks.data_integrity import ConflictingLabels
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_label_ambiguity():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 2, 2, 3, 4]*100
    }
    dataframe = pd.DataFrame(data)
    dataframe = dataframe.astype({'col1': 'category'})
    ds = Dataset(dataframe, label='label')
    check = ConflictingLabels()

    # Act
    result = check.run(ds)

    # Assert
    assert_that(result.value, has_entries({
        'percent': equal_to(1),
        'samples_indices': instance_of(list)
    }))
    assert_that(result.display[1], has_length(2))


def test_label_ambiguity_empty():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 2, 1, 2, 2, 2]*100,
        'label': [1, 2, 1, 1, 1, 1]*100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ConflictingLabels()

    # Act
    result = check.run(ds)

    # Assert
    assert_that(result.value, has_entries({
        'percent': equal_to(0),
        'samples_indices': has_length(0)
    }))
    assert_that(result.display, has_length(0))


def test_label_ambiguity_mixed():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 1, 1, 2, 1]*100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ConflictingLabels()
    # Act
    result = check.run(ds)
    # Assert
    assert_that(result.value, has_entries({
        'percent': close_to(0.5, 0.01),
        'samples_indices': has_length(1)
    }))
    assert_that(
        result.display[1],
        has_length(1)
    )


def test_label_ambiguity_mixed_without_display():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 1, 1, 2, 1]*100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ConflictingLabels()
    # Act
    result = check.run(ds, with_display=False)
    # Assert
    assert_that(result.value, has_entries({
        'percent': close_to(0.5, 0.01),
        'samples_indices': has_length(1)
    }))
    assert_that(result.display, has_length(0))


def test_label_ambiguity_condition():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 1, 1, 2, 1]*100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal()

    # Act
    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Ratio of samples with conflicting labels: 50%',
                               name='Ambiguous sample ratio is less or equal to 0%')
    ))


def test_label_ambiguity_condition_pass():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'col2': [1, 1, 1, 2, 2, 2]*100,
        'col3': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 1, 1, 2, 1]*100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(.7)

    # Act
    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details='Ratio of samples with conflicting labels: 50%',
                               name='Ambiguous sample ratio is less or equal to 70%')
    ))


def test_label_ambiguity_single_column():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2]*100,
        'label': [1, 1, 2, 2, 3, 4]*100
    }
    dataframe = pd.DataFrame(data)
    ds = Dataset(dataframe, label='label')
    check = ConflictingLabels()

    # Act
    result = check.run(ds)

    # Assert
    assert_that(result.value, has_entries({
        'percent': equal_to(1),
        'samples_indices': instance_of(list)
    }))
