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
from sklearn.metrics import log_loss, make_scorer, mean_squared_error

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Dataset
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.tabular.metric_utils.additional_classification_metrics import (false_negative_rate_metric,
                                                                               false_positive_rate_metric,
                                                                               true_negative_rate_metric)
from deepchecks.tabular.utils.task_inference import get_all_labels, infer_classes_from_model
from deepchecks.utils.single_sample_metrics import calculate_neg_cross_entropy_per_sample, calculate_neg_mse_per_sample
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


def test_cross_entropy_lending_club(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    probas = clf.predict_proba(test_ds.features_columns)
    eps = 1e-15

    # Act
    score = calculate_neg_cross_entropy_per_sample(test_ds.label_col, probas, eps=eps)
    score_sklearn = log_loss(test_ds.label_col, probas)

    # Assert
    assert_that(score.mean(), close_to(-1 * 0.524, 0.01))
    assert_that(score.mean(), close_to(-1 * score_sklearn, 0.01))


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


def test_regression_metrics(diabetes, diabetes_model):
    ds, _ = diabetes

    # Act & Assert
    r_2_deepchecks_scorer = DeepcheckScorer('R2', model_classes=None, observed_classes=None)
    score_r_2 = r_2_deepchecks_scorer(diabetes_model, ds)
    assert_that(score_r_2, close_to(0.85, 0.01))

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


def test_mse_diabetes(diabetes_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = diabetes_split_dataset_and_model
    preds = clf.predict(test_ds.features_columns)

    # Act
    score = calculate_neg_mse_per_sample(test_ds.label_col, preds)
    score_sklearn = mean_squared_error(test_ds.label_col, preds)

    # Assert
    assert_that(score.mean(), close_to(-1 * 3296, 1))
    assert_that(score.mean(), close_to(-1 * score_sklearn, 0.01))


# Regression tests for deepchecks/deepchecks#2806: ``make_scorer(...,
# needs_proba=True)`` was deprecated in scikit-learn 1.4 and removed in
# 1.6, so on a current sklearn the call raised at module-import time and
# anything that used ``binary_scorers_dict`` / ``multiclass_scorers_dict``
# (e.g. passing ``scorers=['neg_log_loss']`` to a check) failed before
# scoring. The fix is to switch to ``response_method='predict_proba'``
# whenever sklearn>=1.4 is installed.
#
# These tests fail on origin/main when run against sklearn>=1.6 (the
# module import raises ``TypeError: make_scorer() got an unexpected
# keyword argument 'needs_proba'``) and pass on this branch.

def test_neg_log_loss_scorer_kwargs_match_runtime_sklearn():
    """Verify the module-level kwarg switch picks the right knob for the
    installed sklearn — the *single* place where the fix lives."""
    from packaging import version
    from sklearn import __version__ as scikit_version

    from deepchecks.tabular.metric_utils.scorers import _PROBA_SCORER_KWARGS

    if version.parse(scikit_version) >= version.parse('1.4'):
        assert_that(_PROBA_SCORER_KWARGS, is_({'response_method': 'predict_proba'}))
    else:
        assert_that(_PROBA_SCORER_KWARGS, is_({'needs_proba': True}))


def test_neg_log_loss_scorer_constructible():
    """The bug from #2806 manifests at module-import time on sklearn>=1.6;
    asserting the scorer is in the registry doubles as an import-time
    smoke test."""
    from deepchecks.tabular.metric_utils.scorers import _str_to_scorer_dict, binary_scorers_dict

    assert_that('neg_log_loss' in binary_scorers_dict, is_(True))
    assert_that('neg_log_loss' in _str_to_scorer_dict, is_(True))
    assert_that('roc_auc_per_class' in _str_to_scorer_dict, is_(True))


def test_neg_log_loss_scorer_callable_on_classifier():
    """Issue #2806: the user-visible failure is at *call* time, not at
    ``make_scorer`` time — sklearn 1.6+ silently swallows ``needs_proba``
    via ``**kwargs`` in ``make_scorer`` but then forwards it to
    ``log_loss`` when the scorer is invoked, raising
    ``TypeError: got an unexpected keyword argument 'needs_proba'``.
    This test exercises the call path on a trivial in-memory binary
    classifier — no data fixtures required, no upstream dataset download.
    On origin/main with sklearn>=1.6 it raises the TypeError above; on
    this branch it returns a finite negative log-loss."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from deepchecks.tabular.metric_utils.scorers import binary_scorers_dict

    X = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.], [0., 0.5], [1., 0.5]])
    y_binary = np.array([0, 1, 0, 1, 0, 1])
    binary_clf = LogisticRegression().fit(X, y_binary)

    neg_log_loss = binary_scorers_dict['neg_log_loss']
    score = float(neg_log_loss(binary_clf, X, y_binary))

    assert_that(score <= 0.0, is_(True))


def test_roc_auc_per_class_scorer_scores_multiclass(iris_split_dataset_and_model):
    """Issue #2806: the second occurrence of ``needs_proba`` in this module
    was on ``roc_auc_per_class`` in ``multiclass_scorers_dict``. The
    existing ``test_classification_deepchecks_scorers`` integration test
    (in ``train_test_performance_test.py``) already covers the iris-fitted
    score = 0.997 expectation — this test asserts the same observable
    behavior is preserved by going straight through the scorer registry,
    so a regression is caught even if the integration test path changes."""
    from deepchecks.tabular.metric_utils.scorers import multiclass_scorers_dict

    _, test_ds, clf = iris_split_dataset_and_model
    scorer = deepchecks_scorer(multiclass_scorers_dict['roc_auc_per_class'], clf, test_ds)

    score = scorer(clf, test_ds)

    assert_that(score[1], close_to(0.997, 0.01))
