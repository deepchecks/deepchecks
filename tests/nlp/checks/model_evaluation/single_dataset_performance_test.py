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
import numpy as np
from hamcrest import assert_that, calling, close_to, equal_to, has_items, raises

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.nlp.checks.model_evaluation.single_dataset_performance import SingleDatasetPerformance
from tests.base.utils import equal_condition_result


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
    assert_that(result.value.values[1][-1], close_to(1, 0.001))


def test_run_with_scorer_proba_too_many_classes(text_classification_string_class_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(
            text_classification_string_class_dataset_mock,
            probabilities=[[0.1, 0.4, 0.5], [0.9, 0.05, 0.05], [0.9, 0.01, 0.09]]),
        raises(
            ValidationError,
            'Check requires classification probabilities for the "Train" dataset to have 2 columns, '
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


def test_run_default_scorer_string_class_new_cats_in_model_classes(text_classification_string_class_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'],
                       model_classes=['meh', 'wise', 'zz'])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.666, 0.001))
    assert_that(len(result.value['Class'].unique()), equal_to(3))


def test_multilabel_with_incorrect_model_classes(text_multilabel_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance()

    # Assert
    assert_that(calling(check.run).with_args(text_multilabel_classification_dataset_mock,
                                             model_classes=['meh', 'wise']),
                raises(DeepchecksValueError,
                       'Received model_classes of length 2, but data indicates labels of length 3'))


def test_run_with_scorer_multilabel(text_multilabel_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act
    result = check.run(text_multilabel_classification_dataset_mock,
                       predictions=[[0, 0, 1], [1, 0, 1], [0, 1, 0]])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.777, 0.001))


def test_run_with_scorer_multilabel_w_none(multilabel_mock_dataset_and_probabilities):
    # Arrange
    data, probas = multilabel_mock_dataset_and_probabilities
    data = data.copy()
    assert_that(data.is_multi_label_classification(), equal_to(True))
    data._label = np.asarray(list(data._label[:round(len(data._label) / 2)]) + [None] * round(len(data._label) / 2),
                             dtype=object)
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act
    result = check.run(data, probabilities=probas)

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.831, 0.001))


def test_run_with_scorer_multilabel_class_names(text_multilabel_classification_dataset_mock):
    # Arrange
    text_multilabel_classification_dataset_mock_copy = text_multilabel_classification_dataset_mock.copy()
    check = SingleDatasetPerformance(scorers=['f1_per_class'])

    # Act
    result = check.run(text_multilabel_classification_dataset_mock_copy,
                       predictions=[[0, 0, 1], [1, 0, 1], [0, 1, 0]],
                       model_classes=['a', 'b', 'c'])

    # Assert
    assert_that(result.value.values[0][-1], close_to(1.0, 0.001))
    assert_that(result.value.values[0][0], equal_to('a'))


def test_wikiann_data(small_wikiann_train_test_text_data):
    """Temp to test wikiann dataset loads correctly"""
    _, dataset = small_wikiann_train_test_text_data
    check = SingleDatasetPerformance(scorers=['f1_macro'])
    result = check.run(dataset, predictions=list(dataset.label))
    assert_that(result.value.values[0][-1], equal_to(1))


def test_token_classification_with_none(text_token_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    pred_none_specific_token = [['B-PER', 'O', 'O', 'O', 'O'], ['B-PER', 'O', 'O', 'O', 'O', 'B-GEO'],
                                [None, 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    # Act
    result1 = check.run(text_token_classification_dataset_mock,
                        predictions=pred_none_specific_token)
    # Assert
    assert_that(result1.value.values[0][-1], close_to(0.833, 0.001))

    # TODO: Currently adding None in the predictions list is not supported
    # pred_none_whole_label = [['B-PER', 'O', 'O', 'O', 'O'], ['B-PER', 'O', 'O', 'O', 'O', 'B-GEO'], None]
    # result2 = check.run(text_token_classification_dataset_mock,
    #                     predictions=pred_none_whole_label)
    # assert_that(result1.value.values[0][-1], result2.value.values[0][-1])


def test_run_with_scorer_token(text_token_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    correct_predictions = [['B-PER', 'O', 'O', 'O', 'O'], ['B-PER', 'O', 'O', 'B-GEO', 'O', 'B-GEO'],
                           ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    almost_correct_predictions = [['B-PER', 'O', 'O', 'O', 'O'], ['B-PER', 'O', 'O', 'O', 'O', 'B-GEO'],
                                  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

    # Act
    result = check.run(text_token_classification_dataset_mock,
                       predictions=almost_correct_predictions
                       )

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.833, 0.001))

    # Act
    result = check.run(text_token_classification_dataset_mock,
                       predictions=correct_predictions
                       )

    # Assert
    assert_that(result.value.values[0][-1], close_to(1, 0.001))


def test_run_with_scorer_token_per_class(text_token_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['recall_per_class'])

    # Act
    result = check.run(text_token_classification_dataset_mock,
                       predictions=[['B-PER', 'O', 'O', 'O', 'O'],
                                    ['B-PER', 'O', 'O', 'B-GEO', 'O', 'B-DATE'],
                                    ['O', 'O', 'O', 'B-GEO', 'O', 'O', 'O', 'O']],
                       model_classes=['B-DATE', 'B-GEO', 'B-PER']
                       )

    # Assert
    assert_that(result.value.values[0][-1], close_to(0., 0.001))
    assert_that(result.value.values[0][0], equal_to('B-DATE'))
    assert_that(result.value.values[1][-1], close_to(0.5, 0.001))
    assert_that(result.value.values[1][0], equal_to('B-GEO'))
    assert_that(result.value.values[2][-1], close_to(1., 0.001))
    assert_that(result.value.values[2][0], equal_to('B-PER'))


def test_ignore_O_label_in_model_classes(text_token_classification_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['recall_per_class'])

    # Act
    result = check.run(text_token_classification_dataset_mock,
                       predictions=[['B-PER', 'O', 'O', 'O', 'O'],
                                    ['B-PER', 'O', 'O', 'B-GEO', 'O', 'B-DATE'],
                                    ['O', 'O', 'O', 'B-GEO', 'O', 'O', 'O', 'O']],
                       model_classes=['B-DATE', 'B-GEO', 'B-PER', 'O']
                       )

    # Assert
    assert_that(result.value.values[0][-1], close_to(0., 0.001))
    assert_that(result.value.values[0][0], equal_to('B-DATE'))
    assert_that(result.value.values[1][-1], close_to(0.5, 0.001))
    assert_that(result.value.values[1][0], equal_to('B-GEO'))
    assert_that(result.value.values[2][-1], close_to(1., 0.001))
    assert_that(result.value.values[2][0], equal_to('B-PER'))


def test_condition(text_classification_string_class_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance().add_condition_greater_than(0.7)

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'])
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Failed for metrics: [\'F1\', \'Precision\', \'Recall\']',
                               name='Selected metrics scores are greater than 0.7')
    ))


def test_reduce(text_classification_string_class_dataset_mock):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_per_class']).add_condition_greater_than(0.7)

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'])
    reduce_result = result.reduce_output()

    # Assert
    assert_that(reduce_result['f1_meh'], close_to(0.666, 0.001))
    assert_that(reduce_result['f1_wise'], close_to(0.666, 0.001))
