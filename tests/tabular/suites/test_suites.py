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
"""builtin suites tests"""

# pylint: disable=redefined-outer-name
import typing as t
from datetime import datetime
from hamcrest import assert_that, has_length, contains_exactly

import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from deepchecks.tabular import Dataset, suites
from tests.common import get_expected_results_length, validate_suite_result


@pytest.fixture()
def iris(iris_clean) -> t.Tuple[Dataset, Dataset, AdaBoostClassifier]:
    # note: to run classification suite successfully we need to modify iris dataframe
    # according to suite needs
    df = t.cast(pd.DataFrame, iris_clean.frame.copy())
    df['index'] = range(len(df))
    df['date'] = datetime.now()

    train, test = t.cast(
        t.Tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.33, random_state=42)
    )

    train, test = (
        Dataset(train, label='target', datetime_name='date', index_name='index'),
        Dataset(test, label='target', datetime_name='date', index_name='index')
    )

    model = AdaBoostClassifier(random_state=0)
    model.fit(train.data[train.features], train.data[train.label_name])

    return train, test, model


def _test_suite(train=None, test=None, model=None, **kwargs):
    suite = suites.full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
    result = suite.run(train_dataset=train, test_dataset=test, model=model, **kwargs)
    length = get_expected_results_length(suite, dict(train_dataset=train, test_dataset=test, model=model))
    validate_suite_result(result, length)


def test_iris_single_feature_suite(iris_split_dataset_and_model_single_feature):
    train, test, model = iris_split_dataset_and_model_single_feature
    _test_suite(train, test, model)


def test_iris_single_feature_suite_no_model(iris_split_dataset_and_model_single_feature):
    train, test, _ = iris_split_dataset_and_model_single_feature
    _test_suite(train, test)


def test_kiss_dataset_suite(kiss_dataset_and_model):
    train, test, model = kiss_dataset_and_model
    _test_suite(train, test, model)


def test_weird_classification_suite(wierd_classification_dataset_and_model):
    train, test, model = wierd_classification_dataset_and_model
    _test_suite(train, test, model)


def test_weird_regression_suite(wierd_regression_dataset_and_model):
    train, test, model = wierd_regression_dataset_and_model
    _test_suite(train, test, model)


def test_missing_test_classes_suite(missing_test_classes_binary_dataset_and_model):
    train, test, model = missing_test_classes_binary_dataset_and_model
    _test_suite(train, test, model)


def test_iris_suite(iris):
    train, test, model = iris
    _test_suite(train, test, model)


def test_iris_no_display(iris):
    train, test, model = iris
    _test_suite(train, test, model, with_display=False)


def test_iris_no_model(iris):
    train, test, _ = iris
    _test_suite(train, test)


def test_iris_no_test_dataset(iris):
    train, _, model = iris
    _test_suite(train, None, model)


def test_iris_only_model(iris):
    _, _, model = iris
    _test_suite(None, None, model)


def test_adult_dataset_suite(adult_split_dataset_and_model):
    train, test, model = adult_split_dataset_and_model
    _test_suite(train, test, model)


def test_diabetes_dataset_suite(diabetes_split_dataset_and_model):
    train, test, model = diabetes_split_dataset_and_model
    _test_suite(train, test, model)


def test_generic_boost(
        iris_split_dataset_and_model_cat: t.Tuple[Dataset, Dataset, CatBoostClassifier],
        iris_split_dataset_and_model_xgb: t.Tuple[Dataset, Dataset, XGBClassifier],
        iris_split_dataset_and_model_lgbm: t.Tuple[Dataset, Dataset, LGBMClassifier],
        diabetes_split_dataset_and_model_xgb: t.Tuple[Dataset, Dataset, CatBoostRegressor],
        diabetes_split_dataset_and_model_lgbm: t.Tuple[Dataset, Dataset, XGBRegressor],
        diabetes_split_dataset_and_model_cat: t.Tuple[Dataset, Dataset, LGBMRegressor],
):
    iris_cat_train, iris_cat_test, iris_cat_model = iris_split_dataset_and_model_cat
    iris_xgb_train, iris_xgb_test, iris_xgb_model = iris_split_dataset_and_model_xgb
    iris_lgbm_train, iris_lgbm_test, iris_lgbm_model = iris_split_dataset_and_model_lgbm

    diabetes_cat_train, diabetes_cat_test, diabetes_cat_model = diabetes_split_dataset_and_model_cat
    diabetes_xgb_train, diabetes_xgb_test, diabetes_xgb_model = diabetes_split_dataset_and_model_xgb
    diabetes_lgbm_train, diabetes_lgbm_test, diabetes_lgbm_model = diabetes_split_dataset_and_model_lgbm

    suite = suites.full_suite()

    arguments = (
        dict(train_dataset=iris_cat_train, test_dataset=iris_cat_test, model=iris_cat_model),
        dict(train_dataset=iris_xgb_train, test_dataset=iris_xgb_test, model=iris_xgb_model),
        dict(train_dataset=iris_lgbm_train, test_dataset=iris_lgbm_test, model=iris_lgbm_model),
        dict(train_dataset=diabetes_cat_train, test_dataset=diabetes_cat_test, model=diabetes_cat_model),
        dict(train_dataset=diabetes_xgb_train, test_dataset=diabetes_xgb_test, model=diabetes_xgb_model),
        dict(train_dataset=diabetes_lgbm_train, test_dataset=diabetes_lgbm_test, model=diabetes_lgbm_model),
    )

    for args in arguments:
        result = suite.run(**args)
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def test_generic_custom(
        iris_split_dataset_and_model_custom: t.Tuple[Dataset, Dataset, t.Any],
        diabetes_split_dataset_and_model_custom: t.Tuple[Dataset, Dataset, t.Any],
):
    iris_train, iris_test, iris_model = iris_split_dataset_and_model_custom
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model_custom

    suite = suites.full_suite()

    arguments = (
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model),
        dict(train_dataset=diabetes_train, test_dataset=diabetes_test, model=diabetes_model),
    )

    for args in arguments:
        result = suite.run(**args)
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def test_single_dataset(iris_split_dataset_and_model_custom):
    iris_train, iris_test, iris_model = iris_split_dataset_and_model_custom
    suite = suites.full_suite()
    res_train = suite.run(iris_train, iris_test, iris_model, with_display=False, run_single_dataset='Train')
    expected_train_headers = ['Train Test Performance',
                              'Feature Label Correlation Change',
                              'Feature Label Correlation - Train Dataset',
                              'Feature-Feature Correlation - Train Dataset',
                              'Weak Segments Performance - Train Dataset',
                              'ROC Report - Train Dataset',
                              'Train Test Prediction Drift',
                              'Simple Model Comparison',
                              'Unused Features - Train Dataset',
                              'Model Inference Time - Train Dataset',
                              'Datasets Size Comparison',
                              'New Label Train Test',
                              'New Category Train Test',
                              'String Mismatch Comparison',
                              'Train Test Samples Mix',
                              'Train Test Feature Drift',
                              'Train Test Label Drift',
                              'Multivariate Drift',
                              'Single Value in Column - Train Dataset',
                              'Special Characters - Train Dataset',
                              'Mixed Nulls - Train Dataset',
                              'Mixed Data Types - Train Dataset',
                              'String Mismatch - Train Dataset',
                              'Data Duplicates - Train Dataset',
                              'String Length Out Of Bounds - Train Dataset',
                              'Conflicting Labels - Train Dataset',
                              'Confusion Matrix Report - Train Dataset',
                              'Calibration Metric - Train Dataset',
                              'Outlier Sample Detection - Train Dataset',
                              'Regression Error Distribution - Train Dataset',
                              'Boosting Overfit',
                              'Date Train Test Leakage Duplicates',
                              'Date Train Test Leakage Overlap',
                              'Index Train Test Leakage',
                              'Identifier Label Correlation - Train Dataset']

    res_test = suite.run(iris_train, iris_test, iris_model, with_display=False, run_single_dataset='Test')
    res_full = suite.run(iris_train, iris_test, iris_model, with_display=False)
    res_names = [x.get_header() for x in res_train.results]
    assert_that(res_names, contains_exactly(*expected_train_headers))
    assert_that(res_test.results, has_length(35))
    assert_that(res_full.results, has_length(54))


def test_production_suite(iris):
    suite = suites.production_suite('classification', is_comparative=True)
    train, test, model = iris
    result = suite.run(train, test, model)
    assert_that(result.results, has_length(15))
