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
"""Test user utils"""
from hamcrest import assert_that, calling, close_to, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.feature_importance import calculate_feature_importance
from deepchecks.tabular.metric_utils import DeepcheckScorer


def test_calculate_importance(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model

    # Act
    fi = calculate_feature_importance(adaboost, train_ds)

    # Assert
    assert_that(fi.sum(), close_to(1, 0.000001))


def test_calculate_importance_with_kwargs(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model

    # Act
    scorer = DeepcheckScorer('accuracy', [0, 1, 2], [0, 1, 2])
    fi = calculate_feature_importance(adaboost, train_ds, n_repeats=30, mask_high_variance_features=False,
                                      n_samples=10_000, alternative_scorer=scorer)

    # Assert
    assert_that(fi.sum(), close_to(1, 0.000001))


def test_calculate_importance_force_permutation_fail_on_dataframe(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, adaboost = iris_split_dataset_and_model
    df_only_features = train_ds.data.drop(train_ds.label_name, axis=1)

    # Assert
    assert_that(calling(calculate_feature_importance)
                .with_args(adaboost, df_only_features),
                raises(DeepchecksValueError, 'Cannot calculate permutation feature importance on a pandas Dataframe'))
