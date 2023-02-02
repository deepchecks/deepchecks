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
"""Boosting overfit tests."""
from statistics import mean

from hamcrest import assert_that, close_to, greater_than, has_length
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deepchecks.tabular.checks.model_evaluation.boosting_overfit import BoostingOverfit
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_boosting_classifier(iris):
    # Arrange
    train_df, test = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    test = Dataset(test, label='target')

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train.data[train.features], train.data[train.label_name])

    # Act
    result = BoostingOverfit().run(train, test, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.999, 0.001))
    assert_that(mean(test_scores), close_to(0.961, 0.001))


def test_boosting_xgb_classifier(iris_split_dataset_and_model_xgb):
    # Arrange
    train, test, clf = iris_split_dataset_and_model_xgb

    # Act
    result = BoostingOverfit().run(train, test, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.99, 0.001))
    assert_that(mean(test_scores), close_to(0.985, 0.001))
    assert_that(result.display, has_length(greater_than(0)))


def test_boosting_xgb_classifier_without_display(iris_split_dataset_and_model_xgb):
    # Arrange
    train, test, clf = iris_split_dataset_and_model_xgb

    # Act
    result = BoostingOverfit().run(train, test, clf, with_display=False)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.99, 0.001))
    assert_that(mean(test_scores), close_to(0.985, 0.001))
    assert_that(result.display, has_length(0))


def test_boosting_lgbm_classifier(iris_split_dataset_and_model_lgbm):
    # Arrange
    train, test, clf = iris_split_dataset_and_model_lgbm

    # Act
    result = BoostingOverfit().run(train, test, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.972, 0.001))
    assert_that(mean(test_scores), close_to(0.974, 0.001))


def test_boosting_cat_classifier(iris_split_dataset_and_model_cat):
    # Arrange
    train, test, clf = iris_split_dataset_and_model_cat

    # Act
    result = BoostingOverfit().run(train, test, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.991, 0.001))
    assert_that(mean(test_scores), close_to(0.979, 0.001))


def test_boosting_model_is_pipeline(iris):
    # Arrange
    train_df, test = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    test = Dataset(test, label='target')

    pipe = Pipeline([('scaler', StandardScaler()), ('lr', GradientBoostingClassifier(random_state=0))])
    pipe.fit(train.data[train.features], train.data[train.label_name])

    # Act
    result = BoostingOverfit().run(train, test, pipe)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']

    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.999, 0.001))
    assert_that(mean(test_scores), close_to(0.976, 0.001))


def test_boosting_regressor(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes

    # Act
    result = BoostingOverfit().run(train, test, diabetes_model)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(-44.52, 0.01))
    assert_that(mean(test_scores), close_to(-59.35, 0.01))

def test_boosting_regressor_xgb(diabetes_split_dataset_and_model_xgb):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model_xgb

    # Act
    result = BoostingOverfit().run(train, test, model)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(-22.67, 0.01))
    assert_that(mean(test_scores), close_to(-66.99, 0.01))


def test_boosting_regressor_lgbm(diabetes_split_dataset_and_model_lgbm):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model_lgbm

    # Act
    result = BoostingOverfit().run(train, test, model)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(-41.46, 0.01))
    assert_that(mean(test_scores), close_to(-59.87, 0.01))


def test_boosting_regressor_cat(diabetes_split_dataset_and_model_cat):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model_cat

    # Act
    result = BoostingOverfit().run(train, test, model)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(-35.49, 0.01))
    assert_that(mean(test_scores), close_to(-59.04, 0.01))



def test_boosting_classifier_with_metric(iris):
    # Arrange
    train_df, validation_df = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train.data[train.features], train.data[train.label_name])

    # Act
    result = BoostingOverfit(alternative_scorer=('recall', 'recall_micro')).run(train, validation, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.999, 0.001))
    assert_that(mean(test_scores), close_to(0.961, 0.001))


def test_condition_score_decline_not_greater_than_pass(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes
    check = BoostingOverfit().add_condition_test_score_percent_decline_less_than()

    # Act
    condition_result, *_ = check.conditions_decision(check.run(train, validation, diabetes_model))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found score decline of -3.64%',
        name='Test score over iterations is less than 5% from the best score'
    ))


def test_condition_score_percentage_decline_not_greater_than_not_pass(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes
    check = BoostingOverfit().add_condition_test_score_percent_decline_less_than(0.01)

    # Act
    condition_result, *_ = check.conditions_decision(check.run(train, validation, diabetes_model))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Test score over iterations is less than 1% from the best score',
        details='Found score decline of -3.64%'
    ))
