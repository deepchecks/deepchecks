"""Test feature importance utils"""
import pandas as pd
from hamcrest import equal_to, assert_that, calling, raises, close_to, not_none, none
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

from deepchecks.utils.features import calculate_feature_importance, calculate_feature_importance_or_null, \
                                              column_importance_sorter_df, column_importance_sorter_dict
from deepchecks.errors import DeepchecksValueError


def test_adaboost(iris_split_dataset_and_model):
    train_ds, _, adaboost = iris_split_dataset_and_model
    feature_importances = calculate_feature_importance(adaboost, train_ds)
    assert_that(feature_importances.sum(), equal_to(1))


def test_unfitted(iris_dataset):
    clf = AdaBoostClassifier()
    assert_that(calling(calculate_feature_importance).with_args(clf, iris_dataset),
                raises(NotFittedError, 'This AdaBoostClassifier instance is not fitted yet. Call \'fit\' '
                                       'with appropriate arguments before using this estimator.'))


def test_linear_regression(diabetes):
    ds, _ = diabetes
    clf = LinearRegression()
    clf.fit(ds.features_columns(), ds.label_col())
    feature_importances = calculate_feature_importance(clf, ds)
    assert_that(feature_importances.max(), close_to(0.225374532399, 0.0000000001))
    assert_that(feature_importances.sum(), close_to(1, 0.000001))


def test_calculate_importance(iris_labeled_dataset):
    clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=42)
    clf.fit(iris_labeled_dataset.features_columns(), iris_labeled_dataset.label_col())
    feature_importances = calculate_feature_importance(clf, iris_labeled_dataset)
    assert_that(feature_importances.sum(), close_to(1, 0.000001))


def test_bad_dataset_model(iris_random_forest, diabetes):
    ds, _ = diabetes
    assert_that(calling(calculate_feature_importance).with_args(iris_random_forest, ds),
                raises(DeepchecksValueError, 'Got error when trying to predict with model on dataset'))


def test_calculate_or_null(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    feature_importances = calculate_feature_importance_or_null(train.data, clf)
    assert_that(feature_importances, none())


def test_fi_n_top(diabetes_split_dataset_and_model):
    num_values = 5
    train, _, clf = diabetes_split_dataset_and_model
    columns_info = train.show_columns_info()
    feature_importances = calculate_feature_importance_or_null(train, clf)
    assert_that(feature_importances, not_none())

    feature_importances_sorted = list(feature_importances.sort_values(ascending=False).keys())
    feature_importances_sorted.insert(0, 'target')
    feature_importances_sorted = feature_importances_sorted[:num_values]

    sorted_dict = column_importance_sorter_dict(columns_info, train, feature_importances, num_values)
    assert_that(list(sorted_dict.keys()), equal_to(feature_importances_sorted))

    columns_info_df = pd.DataFrame([columns_info.keys(), columns_info.values()]).T
    columns_info_df.columns = ['keys', 'values']
    sorted_df = column_importance_sorter_df(columns_info_df, train, feature_importances, num_values, col='keys')

    assert_that(list(sorted_df['keys']), equal_to(feature_importances_sorted))

    columns_info_df = columns_info_df.set_index('keys')
    sorted_df = column_importance_sorter_df(columns_info_df, train, feature_importances, num_values)

    assert_that(list(sorted_df.index), equal_to(feature_importances_sorted))
