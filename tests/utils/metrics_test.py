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
"""Test metrics utils"""
import pandas as pd
from hamcrest import assert_that, calling, close_to, has_entries, is_, raises
from sklearn.metrics import make_scorer

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Dataset
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.tabular.metric_utils.additional_classification_metrics import (false_negative_rate_metric,
                                                                               false_positive_rate_metric,
                                                                               true_negative_rate_metric)
from deepchecks.tabular.utils.task_inference import infer_classes_from_model, get_all_labels
from tests.common import is_nan


def deepchecks_scorer(scorer, clf, dataset):
    model_classes = infer_classes_from_model(clf)
    labels = get_all_labels(clf, dataset)
    observed_classes = sorted(labels.unique().tolist())
    return DeepcheckScorer(scorer, model_classes, observed_classes)


def test_lending_club_false_positive_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = make_scorer(false_positive_rate_metric, averaging_method='binary')
    scorer = deepchecks_scorer(binary, clf, test_ds)

    # Act
    score = scorer(clf, test_ds)

    # Assert
    assert_that(score, close_to(0.232, 0.01))


def test_iris_false_positive_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = deepchecks_scorer(make_scorer(false_positive_rate_metric, averaging_method='per_class'), clf, test_ds)
    macro = deepchecks_scorer(make_scorer(false_positive_rate_metric, averaging_method='macro'), clf, test_ds)
    micro = deepchecks_scorer(make_scorer(false_positive_rate_metric, averaging_method='micro'), clf, test_ds)
    weighted = deepchecks_scorer(make_scorer(false_positive_rate_metric, averaging_method='weighted'), clf, test_ds)

    # Act
    score_per_class = per_class(clf, test_ds)
    score_macro = macro(clf, test_ds)
    score_micro = micro(clf, test_ds)
    score_weighted = weighted(clf, test_ds)

    # Assert
    assert_that(score_per_class[0], close_to(0.0, 0))
    assert_that(score_per_class[1], close_to(0.21, 0.01))
    assert_that(score_per_class[2], close_to(0.0, 0))
    assert_that(sum(score_per_class.values()) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.08, 0.01))
    assert_that(score_weighted, close_to(0.063, 0.01))


def test_lending_club_false_negative_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = make_scorer(false_negative_rate_metric, averaging_method='binary')
    scorer = deepchecks_scorer(binary, clf, test_ds)

    # Act
    score = scorer(clf, test_ds)

    # Assert
    assert_that(score, close_to(0.4906, 0.01))


def test_iris_false_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = deepchecks_scorer(make_scorer(false_negative_rate_metric, averaging_method='per_class'), clf, test_ds)
    macro = deepchecks_scorer(make_scorer(false_negative_rate_metric, averaging_method='macro'), clf, test_ds)
    micro = deepchecks_scorer(make_scorer(false_negative_rate_metric, averaging_method='micro'), clf, test_ds)
    weighted = deepchecks_scorer(make_scorer(false_negative_rate_metric, averaging_method='weighted'), clf, test_ds)

    # Act
    score_per_class = per_class(clf, test_ds)
    score_macro = macro(clf, test_ds)
    score_micro = micro(clf, test_ds)
    score_weighted = weighted(clf, test_ds)

    # Assert
    assert_that(score_per_class[0], close_to(0.0, 0))
    assert_that(score_per_class[1], close_to(0, 0.01))
    assert_that(score_per_class[2], close_to(0.105, 0.01))
    assert_that(sum(score_per_class.values()) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.04, 0.01))
    assert_that(score_weighted, close_to(0.033, 0.01))


def test_lending_club_true_negative_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = make_scorer(true_negative_rate_metric, averaging_method='binary')
    scorer = deepchecks_scorer(binary, clf, test_ds)

    # Act
    score = scorer(clf, test_ds)

    # Assert
    assert_that(score, close_to(0.767, 0.01))


def test_iris_true_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = deepchecks_scorer(make_scorer(true_negative_rate_metric, averaging_method='per_class'), clf, test_ds)
    macro = deepchecks_scorer(make_scorer(true_negative_rate_metric, averaging_method='macro'), clf, test_ds)
    micro = deepchecks_scorer(make_scorer(true_negative_rate_metric, averaging_method='micro'), clf, test_ds)
    weighted = deepchecks_scorer(make_scorer(true_negative_rate_metric, averaging_method='weighted'), clf, test_ds)

    # Act
    score_per_class = per_class(clf, test_ds)
    score_macro = macro(clf, test_ds)
    score_micro = micro(clf, test_ds)
    score_weighted = weighted(clf, test_ds)

    # Assert
    assert_that(score_per_class[0], close_to(1, 0))
    assert_that(score_per_class[1], close_to(0.789, 0.01))
    assert_that(score_per_class[2], close_to(1, 0.01))
    assert_that(sum(score_per_class.values()) / 3, close_to(score_macro, 0.00001))
    assert_that(score_micro, close_to(0.92, 0.01))
    assert_that(score_weighted, close_to(0.936, 0.01))


def test_auc_on_regression_task_raises_error(diabetes, diabetes_model):
    ds, _ = diabetes

    # Act & Assert
    auc_deepchecks_scorer = DeepcheckScorer('roc_auc', model_classes=None, observed_classes=None)

    assert_that(calling(auc_deepchecks_scorer).with_args(diabetes_model, ds),
                raises(DeepchecksValueError,
                       'Can\'t compute scorer '
                       r'make_scorer\(roc_auc_score, needs_threshold=True\) when predicted '
                       'probabilities are not provided. Please use a model with predict_proba method or manually '
                       r'provide predicted probabilities to the check\.'))

    auc_deepchecks_scorer = DeepcheckScorer('roc_auc_ovo', model_classes=None, observed_classes=None)

    assert_that(calling(auc_deepchecks_scorer).with_args(diabetes_model, ds),
                raises(DeepchecksValueError,
                       'Can\'t compute scorer '
                       r'make_scorer\(roc_auc_score, needs_proba=True, multi_class=ovo\) when predicted '
                       'probabilities are not provided. Please use a model with predict_proba method or manually '
                       r'provide predicted probabilities to the check\.'))


def test_scorer_with_new_labels(iris: pd.DataFrame, iris_adaboost):
    # Arrange
    iris = iris.copy()
    iris.loc[:10, 'target'] = 19
    iris.loc[10:20, 'target'] = 20
    ds = Dataset(iris, label='target', cat_features=[])
    scorer = deepchecks_scorer('precision_per_class', iris_adaboost, ds)

    # Act
    score = scorer(iris_adaboost, ds)
    # Assert
    assert_that(score, has_entries({
        0: close_to(.58, 0.1), 1: close_to(.92, 0.1), 2: close_to(.95, 0.1), 19: is_nan(), 20: is_nan()
    }))


def test_scorer_with_only_new_labels_in_data(iris: pd.DataFrame, iris_adaboost):
    # Arrange
    iris = iris.copy()
    iris.loc[:50, 'target'] = 19
    iris.loc[50:, 'target'] = 20
    ds = Dataset(iris, label='target', cat_features=[])
    scorer = deepchecks_scorer('precision_per_class', iris_adaboost, ds)

    # Act
    score = scorer(iris_adaboost, ds)
    # Assert
    assert_that(score, has_entries({
        0: is_(0), 1: is_(0), 2: is_(0), 19: is_nan(), 20: is_nan()
    }))
