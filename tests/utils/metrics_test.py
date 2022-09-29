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
from hamcrest import assert_that, close_to, calling, raises
from sklearn.metrics import make_scorer

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.tabular.metric_utils.additional_classification_metrics import (false_negative_rate_metric,
                                                                               false_positive_rate_metric,
                                                                               true_negative_rate_metric)


def test_lending_club_false_positive_rate_scorer_binary(lending_club_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = lending_club_split_dataset_and_model
    binary = make_scorer(false_positive_rate_metric, averaging_method='binary')

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.232, 0.01))


def test_iris_false_positive_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = make_scorer(false_positive_rate_metric, averaging_method='per_class')
    macro = make_scorer(false_positive_rate_metric, averaging_method='macro')
    micro = make_scorer(false_positive_rate_metric, averaging_method='micro')
    weighted = make_scorer(false_positive_rate_metric, averaging_method='weighted')

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
    binary = make_scorer(false_negative_rate_metric, averaging_method='binary')

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.4906, 0.01))


def test_iris_false_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = make_scorer(false_negative_rate_metric, averaging_method='per_class')
    macro = make_scorer(false_negative_rate_metric, averaging_method='macro')
    micro = make_scorer(false_negative_rate_metric, averaging_method='micro')
    weighted = make_scorer(false_negative_rate_metric, averaging_method='weighted')

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
    binary = make_scorer(true_negative_rate_metric, averaging_method='binary')

    # Act
    score = binary(clf, test_ds.features_columns, test_ds.label_col)

    # Assert
    assert_that(score, close_to(0.767, 0.01))


def test_iris_true_negative_rate_scorer_multiclass(iris_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_split_dataset_and_model
    per_class = make_scorer(true_negative_rate_metric, averaging_method='per_class')
    macro = make_scorer(true_negative_rate_metric, averaging_method='macro')
    micro = make_scorer(true_negative_rate_metric, averaging_method='micro')
    weighted = make_scorer(true_negative_rate_metric, averaging_method='weighted')

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


def test_auc_on_regression_task_raises_error(diabetes, diabetes_model):
    ds, _ = diabetes

    # Act & Assert
    auc_deepchecks_scorer = DeepcheckScorer('roc_auc', possible_classes=[0, 1, 2])

    assert_that(calling(auc_deepchecks_scorer).with_args(diabetes_model, ds),
                raises(DeepchecksValueError,
                       'Can\'t compute scorer '
                       r'make_scorer\(roc_auc_score, needs_threshold=True\) when predicted '
                       'probabilities are not provided. Please use a model with predict_proba method or manually '
                       r'provide predicted probabilities to the check\.'))

    auc_deepchecks_scorer = DeepcheckScorer('roc_auc_ovo', possible_classes=[0, 1, 2])

    assert_that(calling(auc_deepchecks_scorer).with_args(diabetes_model, ds),
                raises(DeepchecksValueError,
                       'Can\'t compute scorer '
                       r'make_scorer\(roc_auc_score, needs_proba=True, multi_class=ovo\) when predicted '
                       'probabilities are not provided. Please use a model with predict_proba method or manually '
                       r'provide predicted probabilities to the check\.'))
