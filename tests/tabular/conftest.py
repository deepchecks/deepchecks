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
"""Represents fixtures for unit testing using pytest."""
# pylint: skip-file
import logging
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from hamcrest import any_of, assert_that, instance_of, only_contains
from hamcrest.core.matcher import Matcher
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier, XGBRegressor

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import SingleDatasetBaseCheck
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksBaseError
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.datasets.regression import avocado
from deepchecks.utils.logger import set_verbosity

set_verbosity(logging.WARNING)


def get_expected_results_length(suite: BaseSuite, args: Dict):
    num_single = len([c for c in suite.checks.values() if isinstance(c, SingleDatasetBaseCheck)])
    num_others = len(suite.checks.values()) - num_single
    multiply = 0
    if 'train_dataset' in args:
        multiply += 1
    if 'test_dataset' in args:
        multiply += 1
    # If no train and no test (only model) there will be single result of check failure
    if multiply == 0:
        multiply = 1

    return num_single * multiply + num_others


def validate_suite_result(
        result: SuiteResult,
        min_length: int,
        exception_matcher: Optional[Matcher] = None
):
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))
    assert_that(len(result.results) >= min_length)

    exception_matcher = exception_matcher or only_contains(instance_of(DeepchecksBaseError))

    assert_that(result.results, only_contains(any_of(  # type: ignore
        instance_of(CheckFailure),
        instance_of(CheckResult),
    )))

    failures = [
        it.exception
        for it in result.results
        if isinstance(it, CheckFailure)
    ]

    if len(failures) != 0:
        assert_that(failures, matcher=exception_matcher)  # type: ignore

    for check_result in result.results:
        if isinstance(check_result, CheckResult) and check_result.have_conditions():
            for cond in check_result.conditions_results:
                assert_that(cond.category, any_of(ConditionCategory.PASS,
                                                  ConditionCategory.WARN,
                                                  ConditionCategory.FAIL, ))


@pytest.fixture(scope='session')
def empty_df():
    return pd.DataFrame([])


@pytest.fixture(scope='session')
def kiss_dataset_and_model():
    """A small and stupid dataset and model to catch edge cases."""

    def string_to_length(data: pd.DataFrame):
        data = data.copy()
        data['string_feature'] = data['string_feature'].apply(len)
        return data

    def fillna(data: pd.DataFrame):
        data = data.copy()
        data['numeric_feature'] = data['numeric_feature'].fillna(0)
        data['none_column'] = data['none_column'].fillna(0)
        return data

    df = pd.DataFrame(
        {
            'binary_feature': [0, 1, 1, 0, 0, 1],
            'string_feature': ['ahhh', 'no', 'weeee', 'arg', 'eh', 'E'],
            'numeric_feature': pd.array([4, np.nan, 7, 3, 2, np.nan], dtype='Int64'),
            'none_column': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'numeric_label': [3, 1, 5, 2, 1, 1],
        })
    train, test = train_test_split(df, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label='numeric_label', cat_features=['binary_feature'])
    test_ds = Dataset(test, label='numeric_label', cat_features=['binary_feature'])
    clf = Pipeline([('fillna', FunctionTransformer(fillna)),
                    ('lengthifier', FunctionTransformer(string_to_length)),
                    ('clf', AdaBoostClassifier(random_state=0))])
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


def _get_wierd_dataset_and_model(is_classification, seed=42):
    np.random.seed(seed)

    def string_to_length(data: pd.DataFrame):
        data = data.copy()

        def _len_or_val(x):
            if isinstance(x, str):
                return len(x)
            return x

        data['weird_feature'] = data['weird_feature'].apply(_len_or_val)
        return data

    def sum_tuples(data: pd.DataFrame):
        data = data.copy()
        data['tuples'] = data['tuples'].apply(sum)
        return data

    def fillna(data: pd.DataFrame):
        data = data.copy()
        data = data.fillna(0)
        return data

    index_col = shuffle(list(range(500)) + [str(i) for i in range(500)], random_state=42)
    df = pd.DataFrame(
        {
            'index_col': index_col,
            'index_col_again': index_col,
            'binary_feature': np.random.choice([0, 1], size=1000),
            'bool_feature': np.random.choice([True, False], size=1000),
            'fake_bool_feature': np.random.choice([True, False, 0, 1, np.nan],
                                                  p=[0.4, 0.4, 0.1, 0.05, 0.05], size=1000),
            'weird_feature': np.random.choice(np.array([1, 100, 1.0, 'ahh?', 'wee', np.nan, 0],
                                                       dtype='object'), size=1000),
            8: pd.array(np.random.choice([0, 1, 5, 6, np.nan], size=1000), dtype='Int64'),
            'tuples': random.choices([(0, 2), (1, 6, 8), (9, 1), (8, 1, 9, 8)], k=1000),
            'classification_label': np.random.choice([0, 1, 9, 8], size=1000),
            'regression_label': np.random.random_sample(1000),
        }
    )
    if is_classification:
        df.drop(columns='regression_label', inplace=True)
        label = 'classification_label'
        model_to_use = AdaBoostClassifier
    else:
        df.drop(columns='classification_label', inplace=True)
        label = 'regression_label'
        model_to_use = AdaBoostRegressor
    train, test = train_test_split(df, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label=label, index_name='index_col', cat_features=['binary_feature'])
    test_ds = Dataset(test, label=label, index_name='index_col', cat_features=['binary_feature'])
    clf = Pipeline([('fillna', FunctionTransformer(fillna)),
                    ('sum_tuples', FunctionTransformer(sum_tuples)),
                    ('lengthifier', FunctionTransformer(string_to_length)),
                    ('clf', model_to_use(random_state=0))])
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def wierd_classification_dataset_and_model():
    """A big randomized value dataset for classification."""
    return _get_wierd_dataset_and_model(is_classification=True)


@pytest.fixture(scope='session')
def wierd_regression_dataset_and_model():
    """A big randomized value dataset for regression."""
    return _get_wierd_dataset_and_model(is_classification=False)


@pytest.fixture(scope='session')
def diabetes_dataset_no_label(diabetes_df):
    diabetes_df = diabetes_df.drop('target', axis=1)
    return Dataset(diabetes_df)


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model_custom(diabetes, diabetes_model):
    train, test = diabetes

    class MyModel:
        def predict(self, *args, **kwargs):
            return diabetes_model.predict(*args, **kwargs)

        # sklearn scorers in python 3.6 check fit attr

        def fit(self, *args, **kwargs):
            return diabetes_model.fit(*args, **kwargs)

    return train, test, MyModel()


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model_xgb(diabetes):
    train, test = diabetes
    clf = XGBRegressor(random_state=0)
    clf.fit(train.data[train.features], train.data[train.label_name])
    return train, test, clf


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model_lgbm(diabetes):
    train, test = diabetes
    clf = LGBMRegressor(random_state=0)
    clf.fit(train.data[train.features], train.data[train.label_name])
    return train, test, clf


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model_cat(diabetes):
    train, test = diabetes
    clf = CatBoostRegressor(random_state=0)
    clf.fit(train.data[train.features], train.data[train.label_name], verbose=False)
    return train, test, clf


@pytest.fixture(scope='session')
def iris_dataset_no_label(iris):
    """Return Iris dataset as Dataset object."""
    iris = iris.drop('target', axis=1)
    return Dataset(iris)

@pytest.fixture(scope='session')
def iris_dataset_single_class_labeled(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df, label='target')
    return dataset


@pytest.fixture(scope='session')
def adult_split_dataset_and_model() -> Tuple[Dataset, Dataset, Pipeline]:
    """Return Adult train and val datasets and trained RandomForestClassifier model."""
    train_ds, test_ds = adult.load_data(as_train_test=True)
    model = adult.load_fitted_model()
    return train_ds, test_ds, model


@pytest.fixture(scope='session')
def avocado_split_dataset_and_model() -> Tuple[Dataset, Dataset, Pipeline]:
    """Return Adult train and val datasets and trained RandomForestRegressor model."""
    train_ds, test_ds = avocado.load_data(as_train_test=True)
    model = avocado.load_fitted_model()
    return train_ds, test_ds, model


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_custom(iris_split_dataset_and_model) -> Tuple[Dataset, Dataset, Any]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train_ds, test_ds, clf = iris_split_dataset_and_model

    class MyModel:
        def predict(self, *args, **kwargs):
            return clf.predict(*args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return clf.predict_proba(*args, **kwargs)

        # sklearn scorers in python 3.6 check fit attr

        def fit(self, *args, **kwargs):
            return clf.fit(*args, **kwargs)

    return train_ds, test_ds, MyModel()


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_xgb(iris_split_dataset) -> Tuple[Dataset, Dataset, XGBClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train_ds, test_ds = iris_split_dataset
    clf = XGBClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_lgbm(iris_split_dataset) -> Tuple[Dataset, Dataset, LGBMClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train_ds, test_ds = iris_split_dataset
    clf = LGBMClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_cat(iris_split_dataset) -> Tuple[Dataset, Dataset, CatBoostClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train_ds, test_ds = iris_split_dataset
    clf = CatBoostClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col, verbose=False)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
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
def drifted_data_with_nulls(drifted_data) -> Tuple[Dataset, Dataset]:
    train_ds, test_ds = drifted_data
    n_train_nulls = int(train_ds.n_samples / 10)
    n_test_nulls = int(train_ds.n_samples / 20)

    train_ds = train_ds.copy(train_ds.data)
    test_ds = train_ds.copy(test_ds.data)

    train_ds.data.iloc[:n_train_nulls] = None
    test_ds.data.iloc[:n_test_nulls] = None

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

@pytest.fixture(scope='session')
def adult_no_split():
    ds = adult.load_data(as_train_test=False)
    return ds


@pytest.fixture(scope='session')
def df_with_mixed_datatypes_and_missing_values():
    df = pd.DataFrame({
        'cat': [1, 2, 3, 4, 5], 'dog': [0, 9, 8, np.NAN, 7], 'owl': [np.NAN, 6, 5, 4, 3],
        'red': [np.NAN, np.NAN, np.NAN, np.NAN, np.NAN], 'blue': [0, 1, 2, 3, 4], 'green': [0, 0, 0, 0, 0],
        'white': [0.2, 0.5, 0.6, 0.2, -0.1], 'black': [0.1, 0.2, 0.3, 0.4, 0.5],
        'date': [np.datetime64('2019-01-01'), np.datetime64('2019-12-02'), np.datetime64('2019-01-03'),
                 np.datetime64('2019-02-04'), np.datetime64('2019-01-05')],
        'target': [0, 1, 0, 1, 0]
    }, index=['a', 'b', 'c', 'd', 'e'])
    return df

@pytest.fixture(scope='session')
def missing_test_classes_binary_dataset_and_model():
    """A big randomized value dataset for binary."""
    train_ds, test_ds, clf = _get_wierd_dataset_and_model(is_classification=True)
    test_ds.data[test_ds.label_name] = test_ds.data[test_ds.label_name].apply(lambda x: x % 2)
    return train_ds, test_ds, clf