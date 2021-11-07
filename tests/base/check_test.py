
from hamcrest import assert_that, has_property, has_key, contains_exactly, calling, raises, has_length, has_entries, \
    all_of, equal_to

from mlchecks import BaseCheck, ConditionResult, CheckResult, ConditionCategory
from mlchecks.utils import MLChecksValueError


class DummyCheck(BaseCheck):
    pass


def test_add_condition():
    # Arrange & Act
    check = DummyCheck().add_condition('condition A', lambda r: True)

    # Assert
    assert_that(check, has_property('_conditions', has_key('condition A')))


def test_add_multiple_conditions():
    # Arrange & Act
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: False)
             .add_condition('condition C', lambda r: ConditionResult(True)))

    # Assert
    assert_that(check._conditions.keys(), contains_exactly('condition A', 'condition B', 'condition C'))


def test_add_conditions_wrong_name():
    # Arrange
    check = DummyCheck()

    # Act & Assert
    assert_that(calling(check.add_condition).with_args(333, lambda r: True),
                raises(MLChecksValueError, 'Condition name must be of type str but got: int'))


def test_add_conditions_wrong_value():
    # Arrange
    check = DummyCheck()

    # Act & Assert
    assert_that(calling(check.add_condition).with_args('cond', 'string not function'),
                raises(MLChecksValueError, 'Condition must be a function'))


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
    assert_that(check._conditions.keys(), contains_exactly('condition A', 'condition C'))
    check.remove_condition(0)
    assert_that(check._conditions.keys(), contains_exactly('condition C'))


def test_remove_condition_index_error():
    # Arrange
    check = DummyCheck().add_condition('a', lambda r: True).add_condition('b', lambda r: True)

    # Act & Assert
    assert_that(calling(check.remove_condition).with_args(7),
                raises(MLChecksValueError, 'Index 7 of conditions does not exists'))


def test_condition_decision():
    # Arrange
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: ConditionResult(False, 'some result'))
             .add_condition('condition C', lambda r: ConditionResult(False, 'my actual', ConditionCategory.INSIGHT)))

    decisions = check.conditions_decision(CheckResult(1))

    assert_that(decisions, has_entries({
        'condition A': all_of(
            has_property('is_pass', equal_to(True)),
            has_property('category', ConditionCategory.FAILURE),
            has_property('actual', '')
        ),
        'condition B': all_of(
            has_property('is_pass', equal_to(False)),
            has_property('category', ConditionCategory.FAILURE),
            has_property('actual', 'some result')
        ),
        'condition C': all_of(
            has_property('is_pass', equal_to(False)),
            has_property('category', ConditionCategory.INSIGHT),
            has_property('actual', 'my actual')
        )
    }))
