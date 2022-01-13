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
"""Test metrics utils"""
from hamcrest import equal_to, assert_that, calling, raises
from sklearn.svm import SVC

from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.metrics import task_type_check, ModelType


def test_task_type_check_binary(iris_dataset_single_class, iris_random_forest_single_class):

    res = task_type_check(iris_random_forest_single_class, iris_dataset_single_class)

    assert_that(res, equal_to(ModelType.BINARY))


def test_task_type_check_multiclass(iris_split_dataset_and_model_rf):

    train_ds, _, clf = iris_split_dataset_and_model_rf

    res = task_type_check(clf, train_ds)

    assert_that(res, equal_to(ModelType.MULTICLASS))


def test_task_type_check_regression(diabetes, diabetes_model):

    train_ds, _, = diabetes

    res = task_type_check(diabetes_model, train_ds)

    assert_that(res, equal_to(ModelType.REGRESSION))


def test_task_type_not_sklearn_regression(diabetes):
    class RegressionModel:
        def predict(self, x):
            return [0] * len(x)

    train_ds, _, = diabetes

    res = task_type_check(RegressionModel(), train_ds)

    assert_that(res, equal_to(ModelType.REGRESSION))


def test_task_type_not_sklearn_binary(iris_dataset_single_class):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = task_type_check(ClassificationModel(), iris_dataset_single_class)

    assert_that(res, equal_to(ModelType.BINARY))


def test_task_type_not_sklearn_multiclass(iris_labeled_dataset):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = task_type_check(ClassificationModel(), iris_labeled_dataset)

    assert_that(res, equal_to(ModelType.MULTICLASS))


def test_task_type_check_class_with_no_proba(iris_dataset_single_class):

    clf = SVC().fit(iris_dataset_single_class.features_columns, iris_dataset_single_class.label_col)

    assert_that(calling(task_type_check).with_args(clf, iris_dataset_single_class),
                raises(DeepchecksValueError,
                       r'Model is a sklearn classification model \(a subclass of ClassifierMixin\), but lacks the'
                       r' predict_proba method. Please train the model with probability=True, or skip \/ ignore this'
                       r' check.'))
