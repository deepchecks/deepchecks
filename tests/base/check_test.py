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
"""Tests for BaseCheck class."""
import numpy as np
# pylint: disable-all
import pandas as pd
from hamcrest import *

from deepchecks import __version__
from deepchecks.core import BaseCheck, CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, TrainTestCheck


class DummyCheck(TrainTestCheck):

    def __init__(self, param1=1, param2=2, n_samples=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.n_samples = n_samples

    def run_logic(self, context):
        return CheckResult(context)


def test_add_condition():
    # Arrange & Act
    check = DummyCheck().add_condition('condition A', lambda r: ConditionCategory.PASS)

    # Assert
    assert_that(check._conditions.values(), contains_exactly(
        has_property('name', 'condition A')
    ))


def test_add_multiple_conditions():
    # Arrange & Act
    check = (DummyCheck().add_condition('condition A', lambda r: True)
             .add_condition('condition B', lambda r: False)
             .add_condition('condition C', lambda r: ConditionResult(ConditionCategory.PASS)))

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
             .add_condition('condition C', lambda r: ConditionResult(ConditionCategory.PASS)))

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
    def raise_(ex):  # just to test error in condition
        raise ex

    # Arrange
    check = (DummyCheck().add_condition('condition A', lambda _: True)
             .add_condition('condition B', lambda _: ConditionResult(ConditionCategory.FAIL, 'some result'))
             .add_condition('condition C', lambda _: ConditionResult(ConditionCategory.WARN, 'my actual'))
             .add_condition('condition F', lambda _: raise_(Exception('fail'))))

    decisions = check.conditions_decision(CheckResult(1))

    # Assert
    assert_that(decisions, has_items(
        all_of(
            has_property('name', 'condition A'),
            has_property('category', ConditionCategory.PASS),
            has_property('details', '')
        ),
        all_of(
            has_property('name', 'condition B'),
            has_property('category', ConditionCategory.FAIL),
            has_property('details', 'some result')
        ),
        all_of(
            has_property('name', 'condition C'),
            has_property('category', ConditionCategory.WARN),
            has_property('details', 'my actual')
        ),
        all_of(
            has_property('name', 'condition F'),
            has_property('category', ConditionCategory.ERROR),
            has_property('details', 'Exception in condition: Exception: fail')
        )
    ))


def test_params():
    # Arrange
    default_check = DummyCheck()
    parameter_check = DummyCheck(param2=5)
    all_param_check = DummyCheck(8, 9, 10)

    # Assert
    assert_that(default_check.params(), equal_to({}))
    assert_that(parameter_check.params(), equal_to({'param2': 5}))
    assert_that(all_param_check.params(), equal_to({'param1': 8, 'param2': 9, 'n_samples': 10}))
    assert_that(default_check.params(show_defaults=True), equal_to({'param1': 1, 'param2': 2, 'n_samples': None}))


def test_config():
    check = DummyCheck(param2=5).config()

    assert_that(check, equal_to({
        'module_name': f'{DummyCheck.__module__}',
        'class_name': 'DummyCheck',
        'version': __version__,
        'params': {'param1': 1, 'param2': 5, 'n_samples': None},
    }))

    assert_that(BaseCheck.from_config(check), instance_of(DummyCheck))


def test_pass_feature_importance_incorrect(iris_split_dataset):
    # Arrange
    check = DummyCheck()
    train, test = iris_split_dataset

    # Act & Assert
    assert_that(calling(check.run).with_args(train, test, feature_importance='wrong type'),
                raises(DeepchecksValueError, 'feature_importance must be a pandas Series'))


def test_pass_feature_importance_correct(iris_split_dataset):
    # Arrange
    check = DummyCheck()
    train, test = iris_split_dataset
    feature_importance = pd.Series(data=np.random.rand(len(train.features)), index=train.features)

    # Act
    result = check.run(train, test, feature_importance=feature_importance)
    context: Context = result.value

    # Assert
    assert_that(context._calculated_importance, is_(True))
    assert_that(context._feature_importance is not None)
