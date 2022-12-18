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
"""Test for the Context & _DummyModel creation process"""
from hamcrest import assert_that, calling, raises, close_to

from deepchecks.core.errors import ValidationError
from deepchecks.nlp import Suite
from deepchecks.nlp.checks import SingleDatasetPerformance, KeywordFrequencyDrift

CLASSIFICATION_ERROR_FORMAT = r'Check requires classification for Train to be ' \
                              r'either a sequence that can be cast to a 1D numpy array of shape' \
                              r' \(n_samples,\), or a sequence of sequences that can be cast to a 2D ' \
                              r'numpy array of shape \(n_samples, n_classes\) for the multilabel case.'


def test_wrong_prediction_format(text_classification_dataset_mock):
    # Arrange
    emtpy_suite = Suite('Empty Suite')

    # Act & Assert
    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_predictions=[0, 0, 1, 1]),
        raises(ValidationError, 'Check requires predictions for Train to have 3 rows, same as dataset')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_predictions=[[0, 1], [1, 1], [0, 0]]),
        raises(ValidationError, CLASSIFICATION_ERROR_FORMAT)
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_probabilities=[[0.3, 0.5, 0.2], [0.3, 0.5, 0.2]]),
        raises(ValidationError, 'Check requires classification probabilities for Train dataset to have 3 rows,'
                                ' same as dataset')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_probabilities=[[1, 1, 1], [0, 0, 0], [0.5, 0.5, 0.5]]),
        raises(ValidationError, 'Check requires classification probabilities for Train dataset to have 2 columns, '
                                'same as the number of classes')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_probabilities=[[1, 1], [0, 0], [0.5, 0.2]]),
        raises(ValidationError, 'Check requires classification probabilities for Train dataset to be probabilities and'
                                ' sum to 1 for each row')
    )

    # Run with no error
    emtpy_suite.run(
        train_dataset=text_classification_dataset_mock,
        train_predictions=[1, 1, 1],
        train_probabilities=[[0.9, 0.1], [1, 0], [0.5, 0.5]])


def test_wrong_multilabel_prediction_format(text_multilabel_classification_dataset_mock):
    # Arrange
    emtpy_suite = Suite('Empty Suite')

    # Act & Assert
    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_predictions=[0, 0, 1, 1]),
        raises(ValidationError, 'Check requires predictions for Train to have 3 rows, same as dataset')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_predictions=[0, 1, 1]),
        raises(ValidationError, CLASSIFICATION_ERROR_FORMAT)
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_predictions=[[0], [0, 1], 1]),
        raises(ValidationError, CLASSIFICATION_ERROR_FORMAT)
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_probabilities=[[0.3, 0.5, 0.2], [0.3, 0.5, 0.2]]),
        raises(ValidationError, 'Check requires classification probabilities for Train dataset to have 3 rows,'
                                ' same as dataset')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_probabilities=[[1, 1], [0, 0], [0.5, 0.5]]),
        raises(ValidationError, 'heck requires classification probabilities for Train dataset to have 3 columns, '
                                'same as the number of classes')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_probabilities=[[1, 1.2, 1], [0, 0, 0.3], [0.5, 0.2, 0.9]]),
        raises(ValidationError, 'Check requires classification probabilities for Train dataset to be between 0 and 1')
    )

    # Run with no error
    emtpy_suite.run(
        train_dataset=text_multilabel_classification_dataset_mock,
        train_predictions=[[1, 1, 0], [0, 0, 1], [1, 1, 1]],
        train_probabilities=[[0.9, 0.8, 0.3], [0.9, 0.8, 0.3], [0.9, 0.8, 0.3]])


def test_wrong_token_prediction_format(text_token_classification_dataset_mock): #TODO: Fix predictions
    # Arrange
    emtpy_suite = Suite('Empty Suite')

    pred_error = r'Check requires token classification for Train to have int indices representing the start ' \
                 r'and end of the token at the second and third entry in the token prediction tuples'

    # Act & Assert
    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[[('B-PER', 'a', 4, 0.9)],
                           [('B-PER', 0, 4, 0.9), ('B-GEO', 14, 20, 0.9), ('B-GEO', 25, 30, 0.8)],
                           []]
    ),
        raises(ValidationError, pred_error)
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[[('B-PER', 1.5, 4, 0.9)],
                           [('B-PER', 0, 4, 0.9), ('B-GEO', 14, 20, 0.9), ('B-GEO', 25, 30, 0.8)],
                           []]
    ),
        raises(ValidationError, pred_error)
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[[('B-PER', 5, 4, 0.9)],
                           [('B-PER', 0, 4, 0.9), ('B-GEO', 14, 20, 0.9), ('B-GEO', 25, 30, 0.8)],
                           []]
    ),
        raises(ValidationError, 'Check requires token classification predictions for Train to have token span'
                                ' start before span end')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[[('B-PER', 1, 4, 0.9)],
                           [('B-PER', 0, 4, 0.9), ('B-GEO', 14, 20, 0.9), ('B-GEO', 25, 30, -0.1)],
                           []]
    ),
        raises(ValidationError, 'Check requires token classification for Train to have probabilities between 0 and 1, '
                                'at the fourth entry in the token prediction tuples')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[[('B-PER', 1, 4)],
                           [('B-PER', 0, 4), ('B-GEO', 14, 20), ('B-GEO', 25, 30)],
                           []]
    ),
        raises(ValidationError, 'Check requires token classification for Train to have 4 entries')
    )

    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_token_classification_dataset_mock,
        train_predictions=[1, 2, 3]
    ),
        raises(ValidationError, 'Check requires token classification for Train to be a sequence of sequences')
    )


def test_sampling(text_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['recall_macro'])

    # Act
    result = check.run(text_classification_dataset_mock,
                       predictions=[0, 1, 1])
    result_sampled = check.run(text_classification_dataset_mock,
                               predictions=[0, 1, 1], n_samples=2)

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.75, 0.001))
    assert_that(result_sampled.value.values[0][-1], close_to(0.25, 0.001))


def test_same_dataset(text_classification_dataset_mock):
    # Arrange
    check = KeywordFrequencyDrift()

    # Act
    result = check.run(text_classification_dataset_mock, text_classification_dataset_mock)

    # Assert
    assert_that(result.value['drift_score'], close_to(0.0, 0.001))
