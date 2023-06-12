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
"""Test for the NLP TextPropertyOutliers check"""

import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, raises

from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.nlp.checks import TextPropertyOutliers
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result
from deepchecks.nlp.text_data import TextData


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = TextPropertyOutliers()
    # Act
    result = check.run(test)

    # Assert
    assert_that(len(result.value['Sentiment']['indices']), equal_to(65))
    assert_that(result.value['Sentiment']['lower_limit'], close_to(-0.90, 0.01))
    assert_that(result.value['Sentiment']['upper_limit'], close_to(0.92, 0.01))

    assert_that(len(result.value['Text Length']['indices']), equal_to(0))
    assert_that(result.value['Text Length']['lower_limit'], equal_to(6))
    assert_that(result.value['Text Length']['upper_limit'], equal_to(160))

    # Categorical property:
    assert_that(len(result.value['Language']['indices']), equal_to(55))
    assert_that(result.value['Language']['lower_limit'], close_to(0.007, 0.001))
    assert_that(result.value['Language']['upper_limit'], equal_to(None))

    # Assert display
    assert_that(len(result.display), equal_to(7))
    assert_that(result.display[5], equal_to('<h5><b>Properties Not Shown:</h5></b>'))

    # Check the table of properties not shown:
    expected_series = \
        pd.Series(('Text Length, Subjectivity, Fluency', 'Average Word Length, % Special Characters'),
                  index=('No outliers found.', 'Outliers found but not shown in graphs (n_show_top=5).'))
    result_series = result.display[6].data['Properties']

    assert_that((expected_series != result_series).sum().sum(), equal_to(0))


def test_tweet_emotion_condition(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = TextPropertyOutliers().add_condition_outlier_ratio_less_or_equal()
    # Act
    result = check.run(test)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(len(result.value['Sentiment']['indices']), equal_to(65))
    assert_that(result.value['Sentiment']['lower_limit'], close_to(-0.90, 0.01))
    assert_that(result.value['Sentiment']['upper_limit'], close_to(0.92, 0.01))

    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=False,
            name='Outlier ratio in all properties is less or equal than 5%',
            details='Found 1 properties with outlier ratios above threshold.</br>'
                    'Property with highest ratio is Toxicity with outlier ratio of 16.43%'
        )  # type: ignore
    )


def test_not_enough_samples(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = TextPropertyOutliers(min_samples=6000)

    assert_that(calling(check.run).with_args(test),
                raises(NotEnoughSamplesError, 'Need at least 6000 non-null samples to calculate outliers.'))
    
def test_non_numeric_values_in_text_properties_outliers(tweet_emotion_train_test_textdata):
    # Arrange
    raw_text = ["This is an example.", "Another example here."] * 6
    labels = ["positive", "negative"] * 6
    task_type = "text_classification"
    text_data = TextData(raw_text=raw_text, label=labels, task_type=task_type)
    text_data.calculate_builtin_properties(include_properties=['Sentences Count', 'Average Word Length'])
    text_data.properties['Sentences Count'].iloc[9] = 'as'
    text_data.properties['Average Word Length'].iloc[9] = 100

    # Act
    check = TextPropertyOutliers()
    result = check.run(text_data)

    # Assert
    assert_that(result.value['Sentences Count']['property_error'], equal_to('Numeric property contains non-numeric values.'))
    assert_that(result.value['Average Word Length']['property_error'], equal_to(None))
