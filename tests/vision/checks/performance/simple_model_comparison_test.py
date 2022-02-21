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
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks.performance.simple_model_comparison import SimpleModelComparison
from deepchecks.vision.utils.classification_formatters import ClassificationPredictionFormatter
from deepchecks.vision.datasets.classification.mnist import mnist_prediction_formatter
from hamcrest import assert_that, close_to, equal_to, is_in, calling, raises


def test_mnist_prior_strategy(mnist_dataset_train, mnist_dataset_test, trained_mnist):
    # Arrange

    check = SimpleModelComparison(strategy='prior', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter))
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.994, 0.05))
    assert_that(first_row['Number of samples'], equal_to(1135))
    assert_that(first_row['Class'], equal_to(1))


def test_mnist_not_exist_strategy(mnist_dataset_train, mnist_dataset_test, trained_mnist):
    check = SimpleModelComparison()
    # Act
    assert_that(
        calling(SimpleModelComparison).with_args(strategy='n', n_to_show=2, show_only='largest'),
        raises(DeepchecksValueError)
    )


def test_mnist_most_frequent(mnist_dataset_train, mnist_dataset_test, trained_mnist):
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter))
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.994, 0.05))
    assert_that(first_row['Number of samples'], equal_to(1135))
    assert_that(first_row['Class'], equal_to(1))


def test_mnist_uniform(mnist_dataset_train, mnist_dataset_test, trained_mnist):
    # Arrange
    check = SimpleModelComparison(strategy='uniform', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter))
    # Assert
    assert_that(len(result.value), equal_to(8))


def test_mnist_stratified(mnist_dataset_train, mnist_dataset_test, trained_mnist):
    # Arrange
    check = SimpleModelComparison(strategy='stratified', n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter))
    # Assert
    assert_that(len(result.value), equal_to(8))