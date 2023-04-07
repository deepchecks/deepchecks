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
"""Tests for evaluation_utils.py."""
from hamcrest import close_to, assert_that

from deepchecks.core.evaluation_utils import evaluate_change_in_performance
from deepchecks.tabular.utils.task_type import TaskType


def test_regression(diabetes):
    # Arrange
    train, test = diabetes
    train_data = train.data
    test_data = test.data
    train_data['sex'] = train_data['sex'].apply(lambda x: 0 if x < 0 else 1)
    test_data['sex'] = test_data['sex'].apply(lambda x: 0 if x < 0 else 1)

    train = train.copy(train_data)
    test = test.copy(test_data)

    # Act
    # Give test as an improved version of the train and see performance improves:
    res = evaluate_change_in_performance(old_train_ds=train, old_test_ds=test, new_train_ds=test, new_test_ds=test,
                                         task_type=TaskType.REGRESSION)
    assert_that(res, close_to(1.14, 0.01))


def test_classification(iris_split_dataset):
    # Arrange
    train, test = iris_split_dataset

    # Act
    # Give test as an improved version of the train and see performance improves:
    res = evaluate_change_in_performance(old_train_ds=train, old_test_ds=test, new_train_ds=test, new_test_ds=test,
                                         task_type=TaskType.MULTICLASS)
    assert_that(res, close_to(0.02, 0.01))
