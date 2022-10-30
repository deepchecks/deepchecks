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
from hamcrest import assert_that, equal_to, has_items, is_

from deepchecks.tabular.utils.task_inference import infer_task_type_and_known_classes
from deepchecks.tabular.utils.task_type import TaskType


def test_infer_task_type_binary(iris_dataset_single_class, iris_random_forest_single_class):
    res, observed_classes, model_classes = infer_task_type_and_known_classes(iris_random_forest_single_class, iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))
    assert_that(observed_classes, has_items(0, 1))
    assert_that(model_classes, has_items(0, 1))


def test_infer_task_type_multiclass(iris_split_dataset_and_model_rf):
    train_ds, _, clf = iris_split_dataset_and_model_rf

    res, observed_classes, model_classes = infer_task_type_and_known_classes(clf, train_ds)

    assert_that(res, equal_to(TaskType.MULTICLASS))
    assert_that(observed_classes, has_items(0, 1, 2))
    assert_that(model_classes, has_items(0, 1, 2))


def test_infer_task_type_regression(diabetes, diabetes_model):
    train_ds, _, = diabetes

    res, observed_classes, model_classes = infer_task_type_and_known_classes(diabetes_model, train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))
    assert_that(observed_classes, is_(None))
    assert_that(model_classes, is_(None))


def test_task_type_not_sklearn_regression(diabetes):
    class RegressionModel:
        def predict(self, x):
            return [0] * len(x)

    train_ds, _, = diabetes

    res, observed_classes, model_classes = infer_task_type_and_known_classes(RegressionModel(), train_ds)

    assert_that(res, equal_to(TaskType.REGRESSION))
    assert_that(observed_classes, is_(None))
    assert_that(model_classes, is_(None))


def test_task_type_not_sklearn_binary(iris_dataset_single_class):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res, observed_classes, model_classes = infer_task_type_and_known_classes(ClassificationModel(), iris_dataset_single_class)

    assert_that(res, equal_to(TaskType.BINARY))
    assert_that(observed_classes, has_items(0, 1))
    assert_that(model_classes, is_(None))


def test_task_type_not_sklearn_multiclass(iris_labeled_dataset):
    class ClassificationModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[1, 0]] * len(x)

    res, observed_classes, model_classes = infer_task_type_and_known_classes(ClassificationModel(), iris_labeled_dataset)

    assert_that(res, equal_to(TaskType.MULTICLASS))
    assert_that(observed_classes, has_items(0, 1, 2))
    assert_that(model_classes, is_(None))
