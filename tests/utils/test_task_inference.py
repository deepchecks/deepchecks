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
"""Test task type inference"""
from hamcrest import assert_that, equal_to

from deepchecks.tabular.utils.task_inference import infer_task_type
from deepchecks.tabular.utils.task_type import TaskType


def test_infer_task_type_binary(iris_dataset_single_class, iris_random_forest_single_class):
    res = infer_task_type(iris_random_forest_single_class, iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))


def test_infer_task_type_multiclass(iris_split_dataset_and_model_rf):
    train_ds, _, clf = iris_split_dataset_and_model_rf

    res = infer_task_type(clf, train_ds)

    assert_that(res, equal_to(TaskType.MULTICLASS))


def test_infer_task_type_regression(diabetes, diabetes_model):
    train_ds, _, = diabetes

    res = infer_task_type(diabetes_model, train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))


def test_task_type_not_sklearn_regression(diabetes):
    class RegressionModel:
        def predict(self, x):
            return [0] * len(x)

    train_ds, _, = diabetes

    res = infer_task_type(RegressionModel(), train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))


def test_task_type_not_sklearn_binary(iris_dataset_single_class):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = infer_task_type(ClassificationModel(), iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))


def test_task_type_not_sklearn_multiclass(iris_labeled_dataset):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res = infer_task_type(ClassificationModel(), iris_labeled_dataset)

    assert_that(res, equal_to(TaskType.MULTICLASS))
