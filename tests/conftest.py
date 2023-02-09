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
import logging
# pylint: skip-file
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from deepchecks.core.check_result import CheckResult
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.tabular.datasets.classification import lending_club
from deepchecks.utils.logger import set_verbosity

set_verbosity(logging.WARNING)


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
def diabetes_df():
    diabetes = load_diabetes(return_X_y=False, as_frame=True).frame
    return diabetes


@pytest.fixture(scope='session')
def diabetes(diabetes_df) -> Tuple[Dataset, Dataset]:
    """Return diabetes dataset split to train and test as Datasets."""
    train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
    train = Dataset(train_df, label='target', cat_features=['sex'])
    test = Dataset(test_df, label='target', cat_features=['sex'])
    return train, test


@pytest.fixture(scope='session')
def diabetes_model(diabetes):
    clf = GradientBoostingRegressor(random_state=0)
    train, _ = diabetes
    clf.fit(train.data[train.features], train.data[train.label_name])
    return clf


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
    return Dataset(iris, label='target')


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
def iris_split_dataset(iris_clean) -> Tuple[Dataset, Dataset]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train, test = train_test_split(iris_clean.frame, test_size=0.33, random_state=42)
    train_ds = Dataset(train, label='target')
    test_ds = Dataset(test, label='target')
    return train_ds, test_ds


@pytest.fixture(scope='session')
def iris_split_dataset_and_model(iris_split_dataset) -> Tuple[Dataset, Dataset, AdaBoostClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train_ds, test_ds = iris_split_dataset
    clf = AdaBoostClassifier(random_state=0)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def lending_club_split_dataset_and_model() -> Tuple[Dataset, Dataset, Pipeline]:
    """Return Adult train and val datasets and trained RandomForestClassifier model."""
    train_ds, test_ds = lending_club.load_data(as_train_test=True)
    model = lending_club.load_fitted_model()
    return train_ds, test_ds, model


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_single_feature(iris_clean) -> Tuple[Dataset, Dataset, AdaBoostClassifier]:
    """Return Iris train and val datasets and trained AdaBoostClassifier model."""
    train, test = train_test_split(iris_clean.frame, test_size=0.33, random_state=42)
    train_ds = Dataset(train[['sepal length (cm)', 'target']], label='target')
    test_ds = Dataset(test[['sepal length (cm)', 'target']], label='target')
    clf = Pipeline([('bin', KBinsDiscretizer()),
                    ('clf', AdaBoostClassifier(random_state=0))])
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def iris_split_dataset_and_model_rf(iris_split_dataset) -> Tuple[Dataset, Dataset, RandomForestClassifier]:
    """Return Iris train and val datasets and trained RF model."""
    train_ds, test_ds = iris_split_dataset
    clf = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=2)
    clf.fit(train_ds.features_columns, train_ds.label_col)
    return train_ds, test_ds, clf


@pytest.fixture(scope='session')
def simple_custom_plt_check():
    class DatasetSizeComparison(TrainTestCheck):
        """Check which compares the sizes of train and test datasets."""

        def run_logic(self, context: Context) -> CheckResult:
            # Check logic
            train_size = context.train.n_samples
            test_size = context.test.n_samples

            # Create the check result value
            sizes = {'Train': train_size, 'Test': test_size}
            sizes_df_for_display = pd.DataFrame(sizes, index=['Size'])

            # Display function of matplotlib graph:
            def graph_display():
                plt.bar(sizes.keys(), sizes.values(), color='green')
                plt.xlabel('Dataset')
                plt.ylabel('Size')
                plt.title('Datasets Size Comparison')

            return CheckResult(sizes, display=[sizes_df_for_display, graph_display])

    return DatasetSizeComparison()
