# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""builtin suites tests"""
#pylint: disable=redefined-outer-name
import typing as t
from datetime import datetime

import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from deepchecks.tabular import Dataset, suites
from tests.conftest import get_expected_results_length, validate_suite_result


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


def test_generic_suite(
    iris: t.Tuple[Dataset, Dataset, AdaBoostClassifier],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
    iris_split_dataset_and_model_single_feature : t.Tuple[Dataset, Dataset, AdaBoostClassifier],
        kiss_dataset_and_model,
):
    iris_train, iris_test, iris_model = iris
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model
    kiss_train, kiss_test, kiss_model = kiss_dataset_and_model
    iris_train_single, iris_test_single, iris_model_single= iris_split_dataset_and_model_single_feature
    suite = suites.full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')

    arguments = (
        dict(train_dataset=iris_train_single, test_dataset=iris_test_single, model=iris_model_single),
        dict(train_dataset=iris_train_single, test_dataset=iris_test_single),
        dict(train_dataset=kiss_train, test_dataset=kiss_test, model=kiss_model),
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model),
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model, with_display=False),
        dict(train_dataset=iris_train, test_dataset=iris_test),
        dict(train_dataset=iris_train, model=iris_model),
        dict(train_dataset=diabetes_train, model=diabetes_model),
        dict(train_dataset=diabetes_train, test_dataset=diabetes_test, model=diabetes_model),
        dict(train_dataset=diabetes_train, test_dataset=diabetes_test, model=diabetes_model, with_display=False),
        dict(model=diabetes_model)
    )

    for args in arguments:
        result = suite.run(**args)
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def test_generic_boost(
    iris_split_dataset_and_model_cat: t.Tuple[Dataset, Dataset, CatBoostClassifier],
    iris_split_dataset_and_model_xgb: t.Tuple[Dataset, Dataset, XGBClassifier],
    iris_split_dataset_and_model_lgbm : t.Tuple[Dataset, Dataset, LGBMClassifier],
    diabetes_split_dataset_and_model_xgb: t.Tuple[Dataset, Dataset, CatBoostRegressor],
    diabetes_split_dataset_and_model_lgbm: t.Tuple[Dataset, Dataset, XGBRegressor],
    diabetes_split_dataset_and_model_cat : t.Tuple[Dataset, Dataset, LGBMRegressor],
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
