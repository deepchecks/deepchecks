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
"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name
from typing import Tuple
import numpy as np
import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from deepchecks import Dataset


@pytest.fixture(scope='session')
def multi_index_dataframe():
    """Return a multi-indexed DataFrame."""
    return pd.DataFrame(
        {
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12],
            'd': [13, 14, 15, 16],
        },
        index=pd.MultiIndex.from_product(
            [['a', 'b'], ['c', 'd']],
            names=['first', 'second'],
        ),
    )


@pytest.fixture(scope='session')
def empty_df():
    return pd.DataFrame([])


@pytest.fixture(scope='session')
def diabetes_df():
    diabetes = load_diabetes(return_X_y=False, as_frame=True).frame
    return diabetes


@pytest.fixture(scope='session')
def diabetes(diabetes_df):
    """Return diabetes dataset splited to train and test as Datasets."""
    train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
    train = Dataset(train_df, label='target', cat_features=['sex'])
    test = Dataset(test_df, label='target', cat_features=['sex'])
    return train, test


@pytest.fixture(scope='session')
def diabetes_model(diabetes):
    clf = GradientBoostingRegressor(random_state=0)
    train, _ = diabetes
    return clf.fit(train.features_columns, train.label_col)


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model(diabetes, diabetes_model):
    train, test = diabetes
    clf = diabetes_model
    return train, test, clf


@pytest.fixture(scope='session')
def iris_clean():
    """Return Iris dataset as DataFrame."""
    iris = load_iris(return_X_y=False, as_frame=True)
    return iris


@pytest.fixture(scope='session')
def iris(iris_clean) -> pd.DataFrame:
    """Return Iris dataset as DataFrame."""
    return iris_clean.frame


@pytest.fixture(scope='session')
def iris_dataset(iris):
    """Return Iris dataset as Dataset object."""
    return Dataset(iris)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    """Return trained AdaBoostClassifier on iris data."""
    clf = AdaBoostClassifier(random_state=0)
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
    clf = RandomForestClassifier(random_state=0)
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_random_forest_single_class(iris):
    """Return trained RandomForestClassifier on iris data modified to a binary label."""
    clf = RandomForestClassifier(random_state=0)
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
    dataset = Dataset(df, label='target')
    return dataset


@pytest.fixture(scope='session')
def iris_dataset_single_class_labeled(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df, label='target')
    return dataset


@pytest.fixture
def iris_split_dataset_and_model(iris_clean) -> Tuple[Dataset, Dataset, AdaBoostClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train, test = train_test_split(iris_clean.frame, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label='target')
    val_ds = Dataset(test, label='target')
    clf = AdaBoostClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, val_ds, clf


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_rf(iris) -> Tuple[Dataset, Dataset, RandomForestClassifier]:
    """Return Iris train and val datasets and trained RF model."""
    train, test = train_test_split(iris, test_size=0.33, random_state=0)
    train_ds = Dataset(train, label='target')
    val_ds = Dataset(test, label='target')
    clf = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=2)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, val_ds, clf


@pytest.fixture
def iris_binary_string_split_dataset_and_model(iris) -> Tuple[Dataset, Dataset, DecisionTreeClassifier]:
    """Return Iris train and test datasets and trained DecisionTreeClassifier model."""
    iris = iris.copy()
    iris.loc[iris['target'] != 2, 'target'] = 'a'
    iris.loc[iris['target'] == 2, 'target'] = 'b'
    train, test = train_test_split(iris, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label='target')
    test_ds = Dataset(test, label='target')
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


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


@pytest.fixture(scope='session')
def drifted_data() -> Tuple[Dataset, Dataset]:
    np.random.seed(42)

    train_data = np.concatenate([np.random.randn(1000, 2),
                                 np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))],
                                axis=1)
    test_data = np.concatenate([np.random.randn(1000, 2),
                                np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))],
                               axis=1)

    df_train = pd.DataFrame(train_data,
                            columns=['numeric_without_drift', 'numeric_with_drift', 'categorical_without_drift',
                                     'categorical_with_drift'])
    df_test = pd.DataFrame(test_data, columns=df_train.columns)

    df_train = df_train.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})
    df_test = df_test.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})

    df_test['numeric_with_drift'] = df_test['numeric_with_drift'].astype('float') + abs(
        np.random.randn(1000)) + np.arange(0, 1, 0.001) * 4
    df_test['categorical_with_drift'] = np.random.choice(a=['apple', 'orange', 'banana', 'lemon'],
                                                         p=[0.5, 0.25, 0.15, 0.1], size=(1000, 1))

    label = np.random.randint(0, 2, size=(df_train.shape[0],))
    df_train['target'] = label
    train_ds = Dataset(df_train, label='target')

    label = np.random.randint(0, 2, size=(df_test.shape[0],))
    df_test['target'] = label
    test_ds = Dataset(df_test, label='target')

    return train_ds, test_ds


@pytest.fixture(scope='session')
def drifted_data_and_model(drifted_data) -> Tuple[Dataset, Dataset, Pipeline]:

    train_ds, test_ds = drifted_data

    model = Pipeline([
        ('handle_cat', ColumnTransformer(
            transformers=[
                ('num', 'passthrough',
                 ['numeric_with_drift', 'numeric_without_drift']),
                ('cat',
                 Pipeline([
                     ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                 ]),
                 ['categorical_with_drift', 'categorical_without_drift'])
            ]
        )),
        ('model', DecisionTreeClassifier(random_state=0, max_depth=2))]
    )

    model.fit(train_ds.features_columns, train_ds.label_col)

    return train_ds, test_ds, model


@pytest.fixture(scope='session')
def non_drifted_classification_label() -> Tuple[Dataset, Dataset]:
    np.random.seed(42)

    train_data = np.concatenate([np.random.randn(1000, 2), np.random.choice(a=[1, 0], p=[0.5, 0.5], size=(1000, 1))],
                                axis=1)
    # Create test_data with drift in label:
    test_data = np.concatenate([np.random.randn(1000, 2), np.random.choice(a=[1, 0], p=[0.45, 0.55], size=(1000, 1))],
                               axis=1)

    df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
    df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])

    train_ds = Dataset(df_train, label='target')
    test_ds = Dataset(df_test, label='target')

    return train_ds, test_ds


@pytest.fixture(scope='session')
def drifted_classification_label() -> Tuple[Dataset, Dataset]:
    np.random.seed(42)

    train_data = np.concatenate([np.random.randn(1000, 2), np.random.choice(a=[1, 0], p=[0.5, 0.5], size=(1000, 1))],
                                axis=1)
    # Create test_data with drift in label:
    test_data = np.concatenate([np.random.randn(1000, 2), np.random.choice(a=[1, 0], p=[0.25, 0.75], size=(1000, 1))],
                               axis=1)

    df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
    df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])

    train_ds = Dataset(df_train, label='target')
    test_ds = Dataset(df_test, label='target')

    return train_ds, test_ds


@pytest.fixture(scope='session')
def drifted_regression_label() -> Tuple[Dataset, Dataset]:
    np.random.seed(42)

    train_data = np.concatenate([np.random.randn(1000, 2), np.random.randn(1000, 1)], axis=1)
    test_data = np.concatenate([np.random.randn(1000, 2), np.random.randn(1000, 1)], axis=1)

    df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
    df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])
    # Create drift in test:
    df_test['target'] = df_test['target'].astype('float') + abs(np.random.randn(1000)) + np.arange(0, 1, 0.001) * 4

    train_ds = Dataset(df_train, label='target')
    test_ds = Dataset(df_test, label='target')

    return train_ds, test_ds

