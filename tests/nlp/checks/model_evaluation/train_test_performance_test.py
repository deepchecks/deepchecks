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
"""Test for the NLP Train-Test Performance check"""
import typing as t

import pandas as pd
from hamcrest import *

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.checks import TrainTestPerformance
from deepchecks.nlp.text_data import TextData


class TestTextClassification:

    def test_check_execution(self, tweet_emotion_train_test_textdata):
        # Arrange
        train, test = tweet_emotion_train_test_textdata
        check = TrainTestPerformance()
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == {"F1", "Precision", "Recall"}
        assert set(result.value["Dataset"]) == {"Train", "Test"}

        n_of_samples_per_class = (
            result.value[['Class', 'Number of samples']]
            .groupby('Class')
            .sum()
            .to_dict()
        )
        expected_n_of_samples_per_class = {
            'Number of samples': {
                'anger': 5874,
                'happiness': 3123,
                'optimism': 1251,
                'sadness': 3711
            }
        }

        assert n_of_samples_per_class == expected_n_of_samples_per_class


class TestMultilableClassification:

    def test_check_execution(self):
        # Arrange
        train = TextData(
            raw_text=['I think therefore I am' for _ in range(20)],
            label=[
                *([0, 0, 1] for _ in range(10)),
                *([0, 1, 1] for _ in range(10))
            ],
            task_type='text_classification'
        )
        test = train.copy()
        check = TrainTestPerformance(min_samples=0)
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == {"F1", "Precision", "Recall"}
        assert set(result.value["Dataset"]) == {"Train", "Test"}

        n_of_samples_per_class = (
            result.value[(result.value["Metric"] == "F1") & (result.value["Dataset"] == "Train")]
            .loc[:, ['Class', 'Number of samples']]
            .groupby('Class')
            .sum()
            .to_dict()
        )
        expected_n_of_samples_per_class = {
            'Number of samples': {0: 0, 1: 10, 2: 20}
        }

        assert n_of_samples_per_class == expected_n_of_samples_per_class

    def test_check_execution_with_model_classes(self):
        train = TextData(
            raw_text=['I think therefore I am' for _ in range(20)],
            label=[
                *([0, 0, 1] for _ in range(10)),
                *([0, 1, 1] for _ in range(10))
            ],
            task_type='text_classification'
        )
        test = train.copy()
        check = TrainTestPerformance()
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
            model_classes=['a', 'b', 'c']
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == {"F1", "Precision", "Recall"}
        assert set(result.value["Dataset"]) == {"Train", "Test"}

        n_of_samples_per_class = (
            result.value[(result.value["Metric"] == "F1") & (result.value["Dataset"] == "Train")]
            .loc[:, ['Class', 'Number of samples']]
            .groupby('Class')
            .sum()
            .to_dict()
        )
        expected_n_of_samples_per_class = {
            'Number of samples': {'a': 0, 'b': 10, 'c': 20}
        }

        assert n_of_samples_per_class == expected_n_of_samples_per_class

    def test_check_execution_with_wrong_model_classes(self):
        train = TextData(
            raw_text=['I think therefore I am' for _ in range(20)],
            label=[
                *([0, 0, 1] for _ in range(10)),
                *([0, 1, 1] for _ in range(10))
            ],
            task_type='text_classification'
        )
        test = train.copy()
        check = TrainTestPerformance()

        # Act & Assert
        assert_that(calling(check.run).with_args(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
            model_classes=['a', 'b', 'c', 'd']),
            raises(DeepchecksValueError, 'Received model_classes of length 4, but data indicates labels of length 3')
        )

    def test_display_params(self):
        # Arrange
        train = TextData(
            raw_text=['I think therefore I am' for _ in range(100)],
            label=[
                *([0, 0, 1] for _ in range(50)),
                *([0, 1, 1] for _ in range(50))
            ],
            task_type='text_classification'
        )
        test = train.copy()
        check = TrainTestPerformance(min_samples=101)
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == {"F1", "Precision", "Recall"}
        assert set(result.value["Dataset"]) == {"Train", "Test"}
        assert result.value["Value"].notna().sum() == 0  # all values are NaNs

        check = TrainTestPerformance(n_top_classes=1, show_classes_by='test_largest')
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == {"F1", "Precision", "Recall"}
        assert set(result.value["Dataset"]) == {"Train", "Test"}
        assert isinstance(result.display[1], pd.DataFrame)
        assert result.display[1]['Classes'].loc['Not shown classes (showing only top 1)'] == '[1]'
        assert result.display[1]['Classes'].loc['Classes without enough samples in either Train or Test'] == '[0]'
        assert result.display[0].data[0]['x'].shape == (1,)  # Make sure x-axis has only 1 class

        assert_that(calling(TrainTestPerformance).with_args(show_classes_by='blabla'),
                    raises(DeepchecksValueError))


class TestTokenClassification:

    def test_check_execution_macro(self, small_wikiann_train_test_text_data):
        # Arrange
        train, test = small_wikiann_train_test_text_data
        scorers = ["recall_macro", "f1_macro"]
        check = TrainTestPerformance(scorers=scorers)
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Metric"]) == set(scorers)
        assert set(result.value["Dataset"]) == {"Train", "Test"}
        assert set(result.value["Value"]) == {1.0}

    def test_check_execution_micro(self, small_wikiann_train_test_text_data):
        # Arrange
        train, test = small_wikiann_train_test_text_data
        check = TrainTestPerformance(min_samples=50)
        # Act
        result = check.run(
            train_dataset=train,
            test_dataset=test,
            train_predictions=list(train.label),
            test_predictions=list(test.label),
        )
        # Assert
        assert isinstance(result.value, pd.DataFrame), type(result.value)
        assert set(result.value["Dataset"]) == {"Train", "Test"}
        assert set(result.value[result.value["Value"].notna()]["Value"]) == {1.0}
        assert set(result.value[(result.value["Value"].notna()) & (result.value["Dataset"] == 'Train')]["Class"])\
               == {'PER', 'ORG'}  # LOC has only 49 samples
