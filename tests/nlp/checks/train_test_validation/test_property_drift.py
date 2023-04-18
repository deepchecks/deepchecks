from hamcrest import *
from deepchecks.nlp.checks import PropertyDrift


def test_property_drift_with_datasets_without_drift(tweet_emotion_train_test_textdata):
    # Arrange
    train, _ = tweet_emotion_train_test_textdata
    check = PropertyDrift().add_condition_drift_score_less_than()
    # Act
    result = check.run(train_dataset=train, test_dataset=train)
    condition_results = check.conditions_decision(result)
    # Assert
    assert len(condition_results) == 1
    assert condition_results[0].is_pass() is True

    assert_that(result, has_entries({
        "Formality": has_entries({
            "Drift score": equal_to(0.0),
            "Method": equal_to("Kolmogorov-Smirnov")
        }),
        # "Language": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Cramer's V")
        # }),
        # "Subjectivity": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Average Word Length": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Text Length": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Max Word Length": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Toxicity": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "% Special Characters": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Sentiment": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Fluency": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
    }))  # type: ignore



def test_property_drift_with_drifted_datasets(tweet_emotion_train_test_textdata):
    # Arrange
    train, test = tweet_emotion_train_test_textdata
    train = train.sample(20, random_state=0)
    test = test.sample(20, random_state=0)
    
    train.calculate_default_properties(
        include_long_calculation_properties=True
    )
    test.calculate_default_properties(
        include_long_calculation_properties=True
    )
    
    check = PropertyDrift().add_condition_drift_score_less_than()
    
    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    condition_results = check.conditions_decision(result)
    
    # Assert
    assert len(condition_results) == 1
    assert condition_results[0].is_pass() is False

    
    assert_that(result, has_entries({
        "Formality": has_entries({
            "Drift score": equal_to(0.4),
            "Method": equal_to("Kolmogorov-Smirnov")
        }),
        # "Language": has_entries({
        #     "Drift score": equal_to(0.0),
        #     "Method": equal_to("Cramer's V")
        # }),
        # "Subjectivity": has_entries({
        #     "Drift score": equal_to(0.15000000000000002),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Average Word Length": has_entries({
        #     "Drift score": equal_to(0.4),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Text Length": has_entries({
        #     "Drift score": equal_to(0.19999999999999996),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Max Word Length": has_entries({
        #     "Drift score": equal_to(0.19999999999999996),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Toxicity": has_entries({
        #     "Drift score": equal_to(0.4),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "% Special Characters": has_entries({
        #     "Drift score": equal_to(0.19999999999999996),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Sentiment": has_entries({
        #     "Drift score": equal_to(0.15000000000000002),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
        # "Fluency": has_entries({
        #     "Drift score": equal_to(0.35000000000000003),
        #     "Method": equal_to("Kolmogorov-Smirnov")
        # }),
    }))  # type: ignore
