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
"""Tests for Mixed Nulls check"""
import numpy as np
import pandas as pd

from hamcrest import assert_that, has_length, has_entry, has_property, equal_to, has_items, all_of, is_, close_to
from deepchecks import Dataset, ConditionCategory
from deepchecks.checks.integrity.label_ambiguity import LabelAmbiguity
from tests.checks.utils import equal_condition_result


def test_label_ambiguity():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2],
        'col2': [1, 1, 1, 2, 2, 2],
        'col3': [1, 1, 1, 2, 2, 2],
        'label': [1, 1, 2, 2, 3, 4]
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = LabelAmbiguity()
    # Act
    result = check.run(ds)
    # Assert
    assert_that(result.value, equal_to(1))
    assert_that(result.display[0], has_length(2))


def test_label_ambiguity_empty():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2],
        'col2': [1, 1, 1, 2, 2, 2],
        'col3': [1, 1, 1, 2, 2, 2],
        'label': [1, 1, 1, 1, 1, 1]
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = LabelAmbiguity()
    # Act
    result = check.run(ds)
    # Assert
    assert_that(result.value, equal_to(0))
    assert_that(result.display, has_length(0))


def test_label_ambiguity_mixed():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2],
        'col2': [1, 1, 1, 2, 2, 2],
        'col3': [1, 1, 1, 2, 2, 2],
        'label': [1, 1, 1, 1, 2, 1]
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = LabelAmbiguity()
    # Act
    result = check.run(ds)
    # Assert
    assert_that(result.value, close_to(0.5, 0.01))
    assert_that(result.display[0], has_length(1))


def test_label_ambiguity_condition():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2],
        'col2': [1, 1, 1, 2, 2, 2],
        'col3': [1, 1, 1, 2, 2, 2],
        'label': [1, 1, 1, 1, 2, 1]
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = LabelAmbiguity().add_condition_ambiguous_sample_ratio_not_greater_than()

    # Act
    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 50.00% samples with multiple labels',
                               name='Ambiguous sample ratio is not greater than 0%')
    ))


def test_label_ambiguity_condition_pass():
    # Arrange
    data = {
        'col1': [1, 1, 1, 2, 2, 2],
        'col2': [1, 1, 1, 2, 2, 2],
        'col3': [1, 1, 1, 2, 2, 2],
        'label': [1, 1, 1, 1, 2, 1]
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = LabelAmbiguity().add_condition_ambiguous_sample_ratio_not_greater_than(.7)

    # Act
    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ambiguous sample ratio is not greater than 70.00%')
    ))
