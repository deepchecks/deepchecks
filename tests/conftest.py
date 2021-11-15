"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name
from typing import Tuple

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

from deepchecks import Dataset


@pytest.fixture(scope='session')
def empty_df():
    return pd.DataFrame([])


@pytest.fixture(scope='session')
def diabetes_df():
    diabetes = load_diabetes(return_X_y=False, as_frame=True).frame
    return diabetes


@pytest.fixture(scope='session')
def diabetes(diabetes_df):
    """Return diabetes dataset splited to train and validation as Datasets."""
    train_df, validation_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
    train = Dataset(train_df, label='target', cat_features=['sex'])
    validation = Dataset(validation_df, label='target', cat_features=['sex'])
    return train, validation


@pytest.fixture(scope='session')
def diabetes_model(diabetes):
    clf = GradientBoostingRegressor()
    train, _ = diabetes
    return clf.fit(train.features_columns(), train.label_col())


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model(diabetes, diabetes_model):
    train, validation = diabetes
    clf = diabetes_model
    return train, validation, clf


@pytest.fixture(scope='session')
def iris_clean():
    """Return Iris dataset as DataFrame."""
    iris = load_iris(return_X_y=False, as_frame=True)
    return iris


@pytest.fixture(scope='session')
def iris(iris_clean):
    """Return Iris dataset as DataFrame."""
    return iris_clean.frame


@pytest.fixture(scope='session')
def iris_dataset(iris):
    """Return Iris dataset as Dataset object."""
    return Dataset(iris)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    """Return trained AdaBoostClassifier on iris data."""
    clf = AdaBoostClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_labeled_dataset(iris):
    """Return Iris dataset as Dataset object with label."""
    return Dataset(iris, label='target')


@pytest.fixture(scope='session')
def iris_random_forest(iris):
    """Return trained RandomForestClassifier on iris data."""
    clf = RandomForestClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_random_forest_single_class(iris):
    """Return trained RandomForestClassifier on iris data modified to a binary label."""
    clf = RandomForestClassifier()
    idx = iris.target != 2
    features = iris.drop('target', axis=1)[idx]
    target = iris.target[idx]
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_dataset_single_class(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df)
    return dataset


@pytest.fixture(scope='session')
def iris_dataset_single_class_labeled(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df, label='target')
    return dataset


@pytest.fixture(scope='session')
def iris_split_dataset_and_model(iris_clean) -> Tuple[Dataset, Dataset, AdaBoostClassifier]:
    """Return Iris train and val datasets and trained RF model."""
    train, test = train_test_split(iris_clean.frame, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label='target')
    val_ds = Dataset(test, label='target')
    clf = AdaBoostClassifier()
    clf.fit(train_ds.features_columns(), train_ds.label_col())
    return train_ds, val_ds, clf


# NaN dataframes:
@pytest.fixture(scope='session')
def df_with_nan_row():
    return pd.DataFrame({
        'col1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
        'col2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]})


@pytest.fixture(scope='session')
def df_with_single_nan_in_col():
    return pd.DataFrame({
        'col1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
        'col2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture(scope='session')
def df_with_single_nans_in_different_rows():
    return pd.DataFrame({
        'col1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
        'col2': [0, 1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10]})


@pytest.fixture(scope='session')
def df_with_fully_nan():
    return pd.DataFrame({
        'col1': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'col2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
