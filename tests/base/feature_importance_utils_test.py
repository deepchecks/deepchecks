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
"""Test feature importance utils"""
import pandas as pd
import pytest
from hamcrest import (any_of, assert_that, calling, close_to, contains_exactly, contains_string, equal_to, has_length,
                      is_, none, not_none, raises)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from deepchecks.core.errors import DeepchecksTimeoutError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.utils.feature_importance import (calculate_feature_importance_or_none,
                                                         column_importance_sorter_df, column_importance_sorter_dict,
                                                         _calculate_feature_importance)
from deepchecks.tabular.utils.task_inference import infer_task_type_by_labels, get_all_labels, infer_classes_from_model, \
    infer_task_type_by_class_number
from deepchecks.tabular.utils.task_type import TaskType


def run_fi_calculation(model, dataset, permutation_kwargs=None, force_permutation=False):
    labels = get_all_labels(model, dataset)
    observed_classes = sorted(labels.dropna().unique().tolist())
    model_classes = infer_classes_from_model(model)
    if dataset and dataset.label_type:
        task_type = dataset.label_type
    elif model_classes:
        task_type = infer_task_type_by_class_number(len(model_classes))
    else:
        task_type = infer_task_type_by_labels(labels)
    return _calculate_feature_importance(model=model, dataset=dataset, model_classes=model_classes,
                                         observed_classes=observed_classes, task_type=task_type,
                                         permutation_kwargs=permutation_kwargs, force_permutation=force_permutation)


def test_adaboost(iris_split_dataset_and_model):
    train_ds, _, adaboost = iris_split_dataset_and_model
    feature_importances, fi_type = run_fi_calculation(adaboost, train_ds)
    assert_that(feature_importances.sum(), equal_to(1))
    assert_that(fi_type, is_('feature_importances_'))


def test_unfitted(iris_dataset):
    clf = AdaBoostClassifier()
    assert_that(calling(_calculate_feature_importance).with_args(clf, iris_dataset, model_classes=None,
                                                                 observed_classes=None, task_type=None),
                raises(ModelValidationError, 'Got error when trying to predict with model on dataset: '
                                             'This AdaBoostClassifier instance is not fitted yet. '
                                             'Call \'fit\' with appropriate arguments before using this estimator.'))


def test_linear_regression(diabetes):
    ds, _ = diabetes
    clf = LinearRegression()
    clf.fit(ds.data[ds.features], ds.data[ds.label_name])
    feature_importances, fi_type = run_fi_calculation(clf, ds)
    assert_that(feature_importances.max(), close_to(0.225374532399, 0.0000000001))
    assert_that(feature_importances.sum(), close_to(1, 0.000001))
    assert_that(fi_type, is_('coef_'))


def test_pipeline(iris_split_dataset_and_model_single_feature):
    _, test_ds, clf = iris_split_dataset_and_model_single_feature
    feature_importances, fi_type = run_fi_calculation(clf, test_ds)
    assert_that(feature_importances['sepal length (cm)'], equal_to(1))  # pylint: disable=e1136
    assert_that(feature_importances, has_length(1))
    assert_that(fi_type, is_('permutation_importance'))
    assert_that(hasattr(clf.steps[-1][1], 'feature_importances_'))


def test_logistic_regression():
    train_df = pd.DataFrame([[23, True], [19, False], [15, False], [5, True]], columns=['age', 'smoking'],
                            index=[0, 1, 2, 3])
    train_y = pd.Series([1, 1, 0, 0])

    logreg = LogisticRegression()
    logreg.fit(train_df, train_y)

    ds_train = Dataset(df=train_df, label=train_y)

    feature_importances, fi_type = run_fi_calculation(logreg, ds_train)
    assert_that(feature_importances.sum(), close_to(1, 0.000001))
    assert_that(fi_type, is_('coef_'))


def test_calculate_importance_when_no_builtin(iris_labeled_dataset, caplog):
    # Arrange
    clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=42)
    clf.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
            iris_labeled_dataset.data[iris_labeled_dataset.label_name])

    # Act
    feature_importances, fi_type = run_fi_calculation(clf, iris_labeled_dataset,
                                                      permutation_kwargs={'timeout': 120})
    assert_that(caplog.records, has_length(1))
    assert_that(caplog.records[0].message, equal_to('Could not find built-in feature importance on the model, '
                                            'using permutation feature importance calculation instead'))

    # Assert
    assert_that(feature_importances.sum(), close_to(1, 0.000001))
    assert_that(fi_type, is_('permutation_importance'))


def test_calculate_importance_when_model_is_pipeline(iris_labeled_dataset, caplog):
    # Arrange
    clf = Pipeline([('model', MLPClassifier(hidden_layer_sizes=(10,), random_state=42))])
    clf.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
            iris_labeled_dataset.data[iris_labeled_dataset.label_name])

    # Act
    feature_importances, fi_type = run_fi_calculation(clf, iris_labeled_dataset,
                                                       permutation_kwargs={'timeout': 120})
    assert_that(caplog.records, has_length(1))
    assert_that(caplog.records[0].message, equal_to('Cannot use model\'s built-in feature importance on a Scikit-learn '
                                            'Pipeline, using permutation feature importance calculation instead'))

    # Assert
    assert_that(feature_importances.sum(), close_to(1, 0.000001))
    assert_that(fi_type, is_('permutation_importance'))


def test_calculate_importance_force_permutation_fail_on_timeout(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model

    # Assert
    assert_that(calling(run_fi_calculation)
                .with_args(adaboost, train_ds, force_permutation=True, permutation_kwargs={'timeout': 0}),
                raises(DeepchecksTimeoutError, 'Skipping permutation importance calculation'))


def test_calculate_importance_force_permutation_fail_on_dataframe(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model
    df_only_features = train_ds.data.drop(train_ds.label_name, axis=1)

    # Assert
    assert_that(calling(_calculate_feature_importance)
                .with_args(adaboost, df_only_features, None, None, None, force_permutation=True),
                raises(DeepchecksValueError, 'Cannot calculate permutation feature importance on a pandas Dataframe'))


def test_calculate_importance_when_no_builtin_and_force_timeout(iris_labeled_dataset):
    # Arrange
    clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=42)
    clf.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
            iris_labeled_dataset.data[iris_labeled_dataset.label_name])

    # Act & Assert
    assert_that(calling(run_fi_calculation)
                .with_args(clf, iris_labeled_dataset, force_permutation=True, permutation_kwargs={'timeout': 0}),
                raises(DeepchecksTimeoutError, 'Skipping permutation importance calculation'))


def test_bad_dataset_model(iris_random_forest, diabetes):
    ds, _ = diabetes
    assert_that(
        calling(_calculate_feature_importance).with_args(iris_random_forest, ds, None, None, None),
        any_of(
            # NOTE:
            # depending on the installed version of the scikit-learn
            # will be raised DeepchecksValueError or ModelValidationError
            raises(
                DeepchecksValueError,
                r'(In order to evaluate model correctness we need not empty dataset with the '
                r'same set of features that was used to fit the model. But function received '
                r'dataset with a different set of features.)'),
            raises(
                ModelValidationError,
                r'Got error when trying to predict with model on dataset:(.*)')
        )
    )


def test_calculate_or_null(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    feature_importances = calculate_feature_importance_or_none(clf, train.data, None, None, TaskType.REGRESSION)
    assert_that(feature_importances, contains_exactly(none(), none()))


def test_fi_n_top(diabetes_split_dataset_and_model):
    num_values = 5
    train, _, clf = diabetes_split_dataset_and_model
    columns_info = train.columns_info
    feature_importances, _ = run_fi_calculation(clf, train)
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

    columns_info_df = pd.DataFrame()
    sorted_df = column_importance_sorter_df(columns_info_df, train, feature_importances, num_values, col='keys')
    assert_that(len(sorted_df), equal_to(0))


def test_no_warning_on_none_model(iris_dataset):
    # Act
    with pytest.warns(None) as warn_record:
        fi = calculate_feature_importance_or_none(None, iris_dataset, None, None, TaskType.MULTICLASS)
    # Assert
    assert_that(fi, none())
    assert_that(warn_record, has_length(0))


def test_permutation_importance_with_nan_labels(iris_split_dataset_and_model, caplog):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model
    train_data = train_ds.data.copy()
    train_data.loc[train_data['target'] != 2, 'target'] = None
    train_ds = train_ds.copy(train_data)

    # Act
    feature_importances, fi_type = run_fi_calculation(adaboost, train_ds, force_permutation=True)
    assert_that(caplog.records, has_length(1))
    assert_that(caplog.records[0].message, contains_string('Calculating permutation feature importance without time limit. '
                                                           'Expected to finish in '))

    # Assert
    assert_that(feature_importances.sum(), close_to(1, 0.0001))
    assert_that(fi_type, is_('permutation_importance'))
