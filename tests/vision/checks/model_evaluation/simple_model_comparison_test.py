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
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_items, has_length, is_in, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import SimpleModelComparison
from tests.base.utils import equal_condition_result


def test_mnist_prior_strategy(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison(strategy='prior', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    first_row = result.value.loc[result.value['Model'] != 'Perfect Model'].sort_values(by='Number of samples',
                                                                                       ascending=False).iloc[0]
    # Assert
    assert_that(result.value, has_length(30))
    assert_that(first_row['Value'], close_to(0.245, 0.05))
    assert_that(first_row['Number of samples'], equal_to(28))
    assert_that(first_row['Class'], equal_to(1))


def test_mnist_not_exist_strategy(mnist_visiondata_train, mnist_visiondata_test):
    # Act
    assert_that(
        calling(SimpleModelComparison).with_args(strategy='n', n_to_show=2, show_only='largest'),
        raises(DeepchecksValueError)
    )


def test_mnist_most_frequent(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    first_row = result.value.loc[result.value['Model'] != 'Perfect Model'].sort_values(by='Number of samples',
                                                                                       ascending=False).iloc[0]
    # Assert
    assert_that(result.value, has_length(30))
    assert_that(first_row['Value'], close_to(0.245, 0.05))
    assert_that(first_row['Number of samples'], equal_to(28))
    assert_that(first_row['Class'], equal_to(1))
    assert_that(result.display, has_length(greater_than(0)))


def test_mnist_most_frequent_without_display(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test, with_display=False)
    first_row = result.value.loc[result.value['Model'] != 'Perfect Model'].sort_values(by='Number of samples',
                                                                                       ascending=False).iloc[0]
    # Assert
    assert_that(result.value, has_length(30))
    assert_that(first_row['Value'], close_to(0.245, 0.05))
    assert_that(first_row['Number of samples'], equal_to(28))
    assert_that(first_row['Class'], equal_to(1))
    assert_that(result.display, has_length(0))


def test_mnist_uniform(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison(strategy='uniform', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    first_row = result.value.loc[result.value['Model'] == 'Simple Model'].sort_values(by='Class').iloc[0]
    # Assert
    assert_that(result.value, has_length(30))
    assert_that(first_row['Value'], close_to(0.156, 0.05))
    assert_that(first_row['Number of samples'], equal_to(17))
    assert_that(first_row['Class'], equal_to(0))


def test_mnist_stratified(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison(strategy='stratified', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    first_row = result.value.loc[result.value['Model'] == 'Simple Model'].sort_values(by='Class').iloc[0]
    # Assert
    assert_that(result.value, has_length(30))
    assert_that(first_row['Class'], equal_to(0))
    assert_that(first_row['Value'], close_to(0.136, 0.05))
    assert_that(first_row['Number of samples'], equal_to(17))


def test_condition_failed_for_multiclass(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison().add_condition_gain_greater_than(0.968)
    # Act X
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model is greater than 96.8%',
            details='Found metrics with gain below threshold: {\'F1\': {\'1\': \'95.09%\'}}')

    ))


def test_condition_pass_for_multiclass_avg(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison().add_condition_gain_greater_than(0.43, average=True)
    # Act X
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            details='Found minimal gain of 98.74% for metric F1',
            name='Model performance gain over simple model is greater than 43%')
    ))


def test_condition_pass_for_multiclass_avg_with_classes(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = SimpleModelComparison().add_condition_gain_greater_than(1, average=True, classes=[4])
    # Act X
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model is greater than 100% for classes [4]',
            details='Found metrics with gain below threshold: {\'F1\': \'98.18%\'}'
        )
    ))
