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
"""Test for the NLP UnderAnnotatedSegments check"""
import numpy as np
from hamcrest import assert_that, calling, close_to, equal_to, has_items, raises

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.nlp.checks import UnderAnnotatedMetaDataSegments, UnderAnnotatedPropertySegments
from tests.base.utils import equal_condition_result


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test = test.copy()
    test._label = np.asarray(list(test._label[:round(len(test._label) / 2)]) + [None] * round(len(test._label) / 2),
                             dtype=object)

    check = UnderAnnotatedPropertySegments().add_condition_segments_annotation_ratio_greater_than(0.5)
    # Act
    result = check.run(test)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details=r'Most under annotated segment has annotation ratio of 31.43%.',
                               name=r'In all segments annotation ratio should be greater than 50%.')
    ))

    assert_that(result.value['avg_score'], close_to(0.5, 0.001))
    assert_that(len(result.value['weak_segments_list']), close_to(33, 1))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.314, 0.01))


def test_tweet_emotion_metadata(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test = test.copy()
    test._label = np.asarray(list(test._label[:round(len(test._label) / 2)]) + [None] * round(len(test._label) / 2),
                             dtype=object)
    check = UnderAnnotatedMetaDataSegments().add_condition_segments_relative_performance_greater_than()
    # Act
    result = check.run(test)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found a segment with annotation ratio of 0.366 in comparison to an average score of 0.5 in sampled data.',
                               name='The relative performance of weakest segment is greater than 80% of average model performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.5, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(5))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.366, 0.01))
    assert_that(result.value['weak_segments_list'].iloc[0, 1], equal_to('user_age'))


def test_tweet_emotion_metadata_interesting_segment(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test = test.copy()
    idx_to_change = test.metadata[(test.metadata['user_age'] > 30) & (test.metadata['user_region'] == 'Europe')].index
    label = test._label.copy().astype(object)
    label[idx_to_change] = None
    test._label = label

    # Act
    result = UnderAnnotatedMetaDataSegments().run(test)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.844, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(6))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0, 0.01))
    assert_that(result.value['weak_segments_list'].iloc[0, 1], equal_to('user_age'))


def test_tweet_emotion_metadata_fully_annotated(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = UnderAnnotatedMetaDataSegments().add_condition_segments_relative_performance_greater_than()

    # Act & Assert
    assert_that(
        calling(check.run).with_args(test),
        raises(DeepchecksProcessError, 'Check was unable to find under annotated segments. This is expected if '
                                       'your data is well annotated. If this is not the case, try increasing '
                                       'n_samples or supply more metadata.')
    )


def test_token_classification_dataset(small_wikiann_train_test_text_data):
    # Arrange
    data, _ = small_wikiann_train_test_text_data
    data = data.copy()
    data._label = np.asarray(list(data._label[:40]) + [None] * 10, dtype=object)
    data.calculate_builtin_properties(include_long_calculation_properties=False)
    check = UnderAnnotatedPropertySegments().add_condition_segments_relative_performance_greater_than()

    # Act
    result = check.run(data)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=False,
            details='Found a segment with annotation ratio of 0.375 in comparison to an '
                    'average score of 0.8 in sampled data.',
            name='The relative performance of weakest segment is greater than 80% of average model '
                 'performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.8, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(23))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.375, 0.01))


def test_multilabel_dataset(multilabel_mock_dataset_and_probabilities):
    # Arrange
    data, _ = multilabel_mock_dataset_and_probabilities
    data = data.copy()
    assert_that(data.is_multi_label_classification(), equal_to(True))
    data._label = np.asarray(list(data._label[:round(len(data._label) / 2)]) + [None] * round(len(data._label) / 2),
                             dtype=object)
    check = UnderAnnotatedMetaDataSegments().add_condition_segments_relative_performance_greater_than()

    # Act
    result = check.run(data)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found a segment with annotation ratio of 0.326 in comparison to an average '
                                       'score of 0.5 in sampled data.',
                               name='The relative performance of weakest segment is greater than 80% of average model '
                                    'performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.5, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(5))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.326, 0.01))
