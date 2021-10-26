"""
Contains unit tests for the dominant_frequency_change check
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from mlchecks.base import Dataset
from mlchecks.utils import MLChecksValueError
from mlchecks.checks.integrity import DominantFrequencyChange, dominant_frequency_change
from hamcrest import assert_that, calling, raises, equal_to


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(dominant_frequency_change).with_args(x, x),
                raises(MLChecksValueError,
                'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_no_leakage(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # Arrange
    check = DominantFrequencyChange()
    # Act X
    result = check.run(dataset=val_ds, baseline_dataset=train_ds).value
    # Assert
    assert_that(result, equal_to(None))

def test_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                features=iris_clean.feature_names,
                label='target')

    test_df = pd.concat([x_test, y_test], axis=1)
    test_df.loc[test_df.index % 2 == 0, 'petal length (cm)'] = 5.1

    validation_dataset = Dataset(test_df,
                features=iris_clean.feature_names,
                label='target')
    # Arrange
    check = DominantFrequencyChange()
    # Act X
    result = check.run(dataset=validation_dataset, baseline_dataset=train_dataset).value.to_dict()
    # Assert
    expected_res = {'Value': {'petal length (cm)': 5.1},
                    'Reference data': {'petal length (cm)': '6 (5.71%)'},
                    'Tested data': {'petal length (cm)': '25 (55.56%)'}}
    assert_that(result, equal_to(expected_res))

def test_show_any(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # those params means any value should be included
    check = DominantFrequencyChange(p_value_threshold=2, dominance_ratio=0, ratio_change_thres=-1)
    # Act
    result = check.run(dataset=val_ds, baseline_dataset=train_ds).value
    # Assert
    assert_that(len(result), equal_to(len(train_ds.features())))

def test_show_none_p_val(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # because of p_val no value should be included
    check = DominantFrequencyChange(p_value_threshold=-1, dominance_ratio=0, ratio_change_thres=-1)
    # Act
    result = check.run(dataset=val_ds, baseline_dataset=train_ds).value
    # Assert
    assert_that(result, equal_to(None))

def test_show_none_dominance_ratio(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # because of dominance_ratio no value should be included
    check = DominantFrequencyChange(p_value_threshold=2,
                                    dominance_ratio=len(train_ds.features()) + 1,
                                    ratio_change_thres=-1)
    # Act
    result = check.run(dataset=val_ds, baseline_dataset=train_ds).value
    # Assert
    assert_that(result, equal_to(None))

def test_show_none_ratio_change_thres(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # because of ratio_change_thres no value should be included
    check = DominantFrequencyChange(p_value_threshold=2, dominance_ratio=0, ratio_change_thres=100)
    # Act
    result = check.run(dataset=val_ds, baseline_dataset=train_ds).value
    # Assert
    assert_that(result, equal_to(None))
