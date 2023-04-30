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
"""Test for the properties module"""
from unittest.mock import patch

import pytest
from hamcrest import assert_that, close_to, equal_to

from deepchecks.nlp.utils.text_properties import calculate_default_properties
from tests.nlp.conftest import tweet_emotion_train_test_textdata

def mock_fn(*args, **kwargs):  # pylint: disable=unused-argument
    return [0] * 20_000


@patch('deepchecks.nlp.utils.text_properties.run_available_kwargs', mock_fn)
def test_calculate_default_properties():
    # Arrange
    raw_text = ['This is a test sentence.'] * 20_000

    match_text = r'Calculating the properties \[\'Toxicity\'\] on a large dataset may take a long time.' \
                 r' Consider using a smaller sample size or running this code on better hardware. Consider using a ' \
                 r'GPU or a similar device to run these properties.'

    # Act
    with pytest.warns(UserWarning,
                      match=match_text):
        result = calculate_default_properties(raw_text, include_properties=['Toxicity'],
                                              include_long_calculation_properties=True)[0]

    # Assert
    assert_that(result, equal_to({'Toxicity': [0] * 20_000}))


def test_skipping_long_calculation_properties(tweet_emotion_train_test_textdata):
    
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_default_properties(test_text, include_properties=['Lexical Density', 'Unique Noun Count'])[0]

    # Assert
    assert_that(result['Lexical Density'][0: 10], equal_to([94.44, 93.75, 100.0, 91.67, 87.5, 100.0, 100.0, 100.0, 91.67, 91.67]))
    assert_that(result['Unique Noun Count'][0: 10], equal_to([9, 2, 3, 3, 4, 10, 4, 2, 7, 5]))