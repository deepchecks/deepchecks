import random
import typing as t

import pandas as pd
from hamcrest import *

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


class TestTokenClassification:

    def test_check_execution(self, small_wikiann: t.Tuple[TextData, TextData]):
        # Arrange
        train, test = small_wikiann
        # TODO:
        # currently 'token_*_per_class' scorers are not supported, see DEE-473
        # threfore, we are using only not avg scorers in test
        scorers = ["token_recall_macro", "token_f1_macro"]
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
