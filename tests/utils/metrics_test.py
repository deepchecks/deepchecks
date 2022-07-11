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
"""Test metrics utils"""
from hamcrest import assert_that, calling, equal_to, raises, close_to
from sklearn.svm import SVC

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.metrics import get_false_positive_rate_scorer_binary, \
    get_false_positive_rate_scorer_per_class, get_false_positive_rate_scorer_macro, \
    get_false_positive_rate_scorer_weighted, get_false_positive_rate_scorer_micro, \
    get_false_negative_rate_scorer_per_class, get_false_negative_rate_scorer_macro, \
    get_false_negative_rate_scorer_micro, get_false_negative_rate_scorer_weighted, \
    get_false_negative_rate_scorer_binary, get_true_negative_rate_scorer_per_class, \
    get_true_negative_rate_scorer_macro, get_true_negative_rate_scorer_micro, \
    get_true_negative_rate_scorer_weighted, get_true_negative_rate_scorer_binary
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.tabular.metric_utils.metrics import task_type_check


def test_task_type_check_binary(iris_dataset_single_class, iris_random_forest_single_class):
    res = task_type_check(iris_random_forest_single_class, iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))


def test_task_type_check_multiclass(iris_split_dataset_and_model_rf):
    train_ds, _, clf = iris_split_dataset_and_model_rf

    res = task_type_check(clf, train_ds)

    assert_that(res, equal_to(TaskType.MULTICLASS))


def test_task_type_check_regression(diabetes, diabetes_model):
    train_ds, _, = diabetes

    res = task_type_check(diabetes_model, train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))


def test_task_type_not_sklearn_regression(diabetes):
    class RegressionModel:
        def predict(self, x):
            return [0] * len(x)

    train_ds, _, = diabetes

    res = task_type_check(RegressionModel(), train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))


def test_task_type_not_sklearn_binary(iris_dataset_single_class):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = task_type_check(ClassificationModel(), iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))


def test_task_type_not_sklearn_multiclass(iris_labeled_dataset):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = task_type_check(ClassificationModel(), iris_labeled_dataset)

    assert_that(res, equal_to(TaskType.MULTICLASS))


def test_task_type_check_class_with_no_proba(iris_dataset_single_class):
    clf = SVC().fit(iris_dataset_single_class.data[iris_dataset_single_class.features],
                    iris_dataset_single_class.data[iris_dataset_single_class.label_name])

    assert_that(calling(task_type_check).with_args(clf, iris_dataset_single_class),
                raises(DeepchecksValueError,
                       r'Model is a sklearn classification model \(a subclass of ClassifierMixin\), but lacks the'
                       r' predict_proba method. Please train the model with probability=True, or skip \/ ignore this'
                       r' check.'))


def test_lending_club_false_positive_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = get_false_positive_rate_scorer_binary()

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.232, 0.01))


def test_iris_false_positive_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = get_false_positive_rate_scorer_per_class()
    macro = get_false_positive_rate_scorer_macro()
    micro = get_false_positive_rate_scorer_micro()
    weighted = get_false_positive_rate_scorer_weighted()

    # Act
    score_per_class = per_class(clf, test_ds.features_columns, test_ds.label_col)
    score_macro = macro(clf, test_ds.features_columns, test_ds.label_col)
    score_micro = micro(clf, test_ds.features_columns, test_ds.label_col)
    score_weighted = weighted(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score_per_class[0], close_to(0.0, 0))
    assert_that(score_per_class[1], close_to(0.21, 0.01))
    assert_that(score_per_class[2], close_to(0.0, 0))
    assert_that(sum(score_per_class) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.08, 0.01))
    assert_that(score_weighted, close_to(0.063, 0.01))


def test_lending_club_false_negative_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = get_false_negative_rate_scorer_binary()

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.4906, 0.01))


def test_iris_false_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = get_false_negative_rate_scorer_per_class()
    macro = get_false_negative_rate_scorer_macro()
    micro = get_false_negative_rate_scorer_micro()
    weighted = get_false_negative_rate_scorer_weighted()

    # Act
    score_per_class = per_class(clf, test_ds.features_columns, test_ds.label_col)
    score_macro = macro(clf, test_ds.features_columns, test_ds.label_col)
    score_micro = micro(clf, test_ds.features_columns, test_ds.label_col)
    score_weighted = weighted(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score_per_class[0], close_to(0.0, 0))
    assert_that(score_per_class[1], close_to(0, 0.01))
    assert_that(score_per_class[2], close_to(0.105, 0.01))
    assert_that(sum(score_per_class) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.04, 0.01))
    assert_that(score_weighted, close_to(0.033, 0.01))


def test_lending_club_true_negative_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = get_true_negative_rate_scorer_binary()

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.767, 0.01))


def test_iris_true_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = get_true_negative_rate_scorer_per_class()
    macro = get_true_negative_rate_scorer_macro()
    micro = get_true_negative_rate_scorer_micro()
    weighted = get_true_negative_rate_scorer_weighted()

    # Act
    score_per_class = per_class(clf, test_ds.features_columns, test_ds.label_col)
    score_macro = macro(clf, test_ds.features_columns, test_ds.label_col)
    score_micro = micro(clf, test_ds.features_columns, test_ds.label_col)
    score_weighted = weighted(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score_per_class[0], close_to(1, 0))
    assert_that(score_per_class[1], close_to(0.789, 0.01))
    assert_that(score_per_class[2], close_to(1, 0.01))
    assert_that(sum(score_per_class) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.92, 0.01))
    assert_that(score_weighted, close_to(0.936, 0.01))
