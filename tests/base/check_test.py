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
"""Tests for BaseCheck class."""
# pylint: disable=protected-access
from hamcrest import assert_that, has_property, contains_exactly, calling, raises, has_length, \
    all_of, equal_to, has_items

from deepchecks import BaseCheck, ConditionResult, CheckResult, ConditionCategory
from deepchecks.errors import DeepchecksValueError


class DummyCheck(BaseCheck):
    pass


def test_add_condition():
    # Arrange & Act
    check = DummyCheck().add_condition('condition A', lambda r: True)

    # Assert
    assert_that(check._conditions.values(), contains_exactly(
        has_property('name', 'condition A')
    ))


def test_add_multiple_conditions():
    # Arrange & Act
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: False)
             .add_condition('condition C', lambda r: ConditionResult(True)))

    # Assert
    assert_that(check._conditions.values(), contains_exactly(
        has_property('name', 'condition A'),
        has_property('name', 'condition B'),
        has_property('name', 'condition C')
    ))


def test_add_conditions_wrong_name():
    # Arrange
    check = DummyCheck()

    # Act & Assert
    assert_that(calling(check.add_condition).with_args(333, lambda r: True),
                raises(DeepchecksValueError, 'Condition name must be of type str but got: int'))


def test_add_conditions_wrong_value():
    # Arrange
    check = DummyCheck()

    # Act & Assert
    assert_that(calling(check.add_condition).with_args('cond', 'string not function'),
                raises(DeepchecksValueError, 'Condition must be a function'))


def test_clean_conditions():
    # Arrange
    check = DummyCheck().add_condition('a', lambda r: True).add_condition('b', lambda r: True)

    # Act & Assert
    assert_that(check, has_property('_conditions', has_length(2)))
    check.clean_conditions()
    assert_that(check, has_property('_conditions', has_length(0)))


def test_remove_condition():
    # Arrange
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: False)
             .add_condition('condition C', lambda r: ConditionResult(True)))

    # Act & Assert
    check.remove_condition(1)
    assert_that(check._conditions.values(), has_items(
        has_property('name', 'condition A'), has_property('name', 'condition C')
    ))
    check.remove_condition(0)
    assert_that(check._conditions.values(), has_items(has_property('name', 'condition C')))


def test_remove_condition_index_error():
    # Arrange
    check = DummyCheck().add_condition('a', lambda r: True).add_condition('b', lambda r: True)

    # Act & Assert
    assert_that(calling(check.remove_condition).with_args(7),
                raises(DeepchecksValueError, 'Index 7 of conditions does not exists'))


def test_condition_decision():
    # Arrange
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: ConditionResult(False, 'some result'))
             .add_condition('condition C', lambda r: ConditionResult(False, 'my actual', ConditionCategory.WARN)))

    decisions = check.conditions_decision(CheckResult(1))

    # Assert
    assert_that(decisions, has_items(
        all_of(
            has_property('name', 'condition A'),
            has_property('is_pass', equal_to(True)),
            has_property('category', ConditionCategory.FAIL),
            has_property('details', '')
        ),
        all_of(
            has_property('name', 'condition B'),
            has_property('is_pass', equal_to(False)),
            has_property('category', ConditionCategory.FAIL),
            has_property('details', 'some result')
        ),
        all_of(
            has_property('name', 'condition C'),
            has_property('is_pass', equal_to(False)),
            has_property('category', ConditionCategory.WARN),
            has_property('details', 'my actual')
        )
    ))
