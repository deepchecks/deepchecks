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
"""Contains unit tests for the tabular package deprecation warnings."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (CategoryMismatchTrainTest, MultiModelPerformanceReport,
                                       BoostingOverfit, RegressionSystematicError, SegmentPerformance,
                                       SimpleModelComparison,
                                       TrainTestFeatureDrift, TrainTestLabelDrift, TrainTestPredictionDrift,
                                       WeakSegmentsPerformance, WholeDatasetDrift)


def test_deprecation_segment_performance_warning():
    with pytest.warns(DeprecationWarning, match='The SegmentPerformance check is deprecated and will be removed in the '
                                                '0.11 version. Please use the WeakSegmentsPerformance check instead.'):
        _ = SegmentPerformance()


def test_deprecation_whole_dataset_drift_warning():
    with pytest.warns(DeprecationWarning, match='The WholeDatasetDrift check is deprecated and will be removed in the '
                                                '0.11 version. Please use the MultivariateDrift check instead.'):
        _ = WholeDatasetDrift()


def test_deprecation_systematic_regression_warning():
    with pytest.warns(DeprecationWarning, match='RegressionSystematicError check is deprecated and will be removed in '
                                                'future version, please use '
                                                'RegressionErrorDistribution check instead.'):
        _ = RegressionSystematicError()


def test_deprecation_label_type_dataset():
    with pytest.warns(DeprecationWarning, match='regression_label value for label type is deprecated, allowed task '
                                                'types are multiclass, binary and regression.'):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        Dataset(df, label='b', label_type='regression_label')


def test_deprecation_y_pred_train_single_dataset():
    ds = Dataset(pd.DataFrame({'a': np.random.randint(0, 5, 50), 'b': np.random.randint(0, 5, 50),
                               'label': np.random.randint(0, 2, 50)}), label='label')
    y_pred_train = np.array(np.random.randint(0, 2, 50))
    y_proba_train = np.random.rand(50, 2)
    with pytest.warns(DeprecationWarning, match='y_pred_train is deprecated, please use y_pred instead.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred_train=y_pred_train, y_proba_train=y_proba_train)

    with pytest.warns(DeprecationWarning, match='y_proba_train is deprecated, please use y_proba instead.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred_train=y_pred_train, y_proba_train=y_proba_train)


def test_deprecation_y_pred_test_single_dataset():
    ds = Dataset(pd.DataFrame({'a': np.random.randint(0, 5, 50), 'b': np.random.randint(0, 5, 50),
                               'label': np.random.randint(0, 2, 50)}), label='label')
    y_pred_train = np.array(np.random.randint(0, 2, 50))
    y_proba_train = np.random.rand(50, 2)
    with pytest.warns(DeprecationWarning, match='y_pred_test is deprecated and ignored.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred=y_pred_train, y_proba=y_proba_train,
                                          y_pred_test=y_pred_train, y_proba_test=y_proba_train)

    with pytest.warns(DeprecationWarning, match='y_proba_test is deprecated and ignored.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred=y_pred_train, y_proba=y_proba_train,
                                          y_pred_test=y_pred_train, y_proba_test=y_proba_train)


def test_deprecation_warning_simple_model_comparison():
    # Test that warning is raised when alternative_scorers has value:
    with pytest.warns(DeprecationWarning, match='alternative_scorers'):
        _ = SimpleModelComparison(alternative_scorers={'acc': accuracy_score})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = SimpleModelComparison()


def test_deprecation_warning_multi_model_performance_report():
    # Test that warning is raised when alternative_scorers has value:
    with pytest.warns(DeprecationWarning, match='alternative_scorers'):
        _ = MultiModelPerformanceReport(alternative_scorers={'acc': accuracy_score})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = MultiModelPerformanceReport()


@pytest.mark.parametrize(
    ('check_class', 'alternative_scorer'),
    [
        (BoostingOverfit, ('Recall', 'recall_micro')),
        (WeakSegmentsPerformance, {'Recall': 'recall_micro'}),
    ],
)
def test_deprecation_warning_alternative_scorer(check_class, alternative_scorer):
    with pytest.warns(DeprecationWarning, match='alternative_scorer'):
        check = check_class(alternative_scorer=alternative_scorer)

    assert check.scorers == {'Recall': 'recall_micro'}


@pytest.mark.parametrize('check_class', [BoostingOverfit, WeakSegmentsPerformance])
def test_scorers_parameter_does_not_raise_warning(check_class):
    scorers = ['recall_micro', 'precision_micro']

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check = check_class(scorers=scorers)

    assert check.scorers == scorers


def test_deprecation_category_mismatch_train_test():
    with pytest.warns(DeprecationWarning, match='CategoryMismatchTrainTest is deprecated, use NewCategoryTrainTest '
                                                'instead'):
        _ = CategoryMismatchTrainTest()


def test_deprecation_warning_train_test_prediction_drift():
    with pytest.warns(DeprecationWarning, match="The TrainTestPredictionDrift check is deprecated and will be removed"
                                                " in the 0.14 version. Please use the PredictionDrift check instead."):
        _ = TrainTestPredictionDrift()


def test_deprecation_warning_train_test_feature_drift():
    with pytest.warns(DeprecationWarning, match="The TrainTestFeatureDrift check is deprecated and will be removed in"
                                                " the 0.14 version. Please use the FeatureDrift check instead"):
        _ = TrainTestFeatureDrift()


def test_deprecation_warning_train_test_label_drift():
    with pytest.warns(DeprecationWarning, match="The TrainTestLabelDrift check is deprecated and will be removed in "
                                                "the 0.14 version.Please use the LabelDrift check instead."):
        _ = TrainTestLabelDrift()
