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
"""Test for the nlp SingleDatasetPerformance check"""

from hamcrest import assert_that, close_to, calling, raises, equal_to

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.nlp.checks.model_evaluation.single_dataset_performance import SingleDatasetPerformance


def test_run_with_scorer(text_classification_dataset_mock):
    """Test that the check runs with a scorer override"""
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act
    result = check.run(text_classification_dataset_mock,
                       predictions=[0, 1, 1])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.666, 0.001))


def test_run_with_scorer_proba(text_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro', 'roc_auc'])

    # Act
    result = check.run(text_classification_dataset_mock,
                       probabilities=[[0.9, 0.1], [0.1, 0.9], [0.05, 0.95]])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.666, 0.001))
    assert_that(result.value.values[1][-1], close_to(0.75, 0.001))


def test_run_with_scorer_proba_too_many_classes(text_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(text_classification_dataset_mock,
                                     probabilities=[[0.1, 0.4, 0.5], [0.9, 0.05, 0.05], [0.9, 0.01, 0.09]]),
        raises(ValidationError, 'Check requires classification probabilities for train dataset to have 2 columns, '
                                'same as the number of classes')
    )


def test_run_with_illegal_scorer(text_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_mean'])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(text_classification_dataset_mock,
                                     predictions=[0, 1, 1]),
        raises(DeepchecksValueError, 'Scorer name f1_mean is unknown. See metric guide for a list'
                                     ' of allowed scorer names.')
    )


def test_run_default_scorer_string_class(text_classification_string_class_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.666, 0.001))


def test_run_with_scorer_multilabel(text_multilabel_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act
    result = check.run(text_multilabel_classification_dataset_mock,
                       predictions=[[0, 0, 1], [1, 0, 1], [0, 1, 0]])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.777, 0.001))


def test_run_with_scorer_multilabel_class_names(text_multilabel_classification_dataset_mock):
    # Arrange
    text_multilabel_classification_dataset_mock_copy = text_multilabel_classification_dataset_mock.copy()
    text_multilabel_classification_dataset_mock_copy._classes = ['a', 'b', 'c']
    check = SingleDatasetPerformance(scorers=['f1_per_class'])

    # Act
    result = check.run(text_multilabel_classification_dataset_mock_copy,
                       predictions=[[0, 0, 1], [1, 0, 1], [0, 1, 0]])

    # Assert
    assert_that(result.value.values[0][-1], close_to(1.0, 0.001))
    assert_that(result.value.values[0][0], equal_to('a'))
