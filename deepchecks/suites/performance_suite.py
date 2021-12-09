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
"""The predefined performance suite module."""
from deepchecks import Suite
from deepchecks.checks import TrustScoreComparison, NewLabelTrainTest
from deepchecks.checks.performance import (
    PerformanceReport,
    ConfusionMatrixReport,
    RocReport,
    CalibrationMetric,
    ClassPerformanceImbalance,
    SimpleModelComparison,
    RegressionSystematicError,
    RegressionErrorDistribution
)

__all__ = [
    'classification_suite',
    'regression_suite',
    'generic_performance_suite',
    'regression_suite',
    'performance_suite'
]


def classification_suite() -> Suite:
    """Create 'Classification Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the classification model.
    """
    return Suite(
        'Classification Suite',
        ConfusionMatrixReport(),
        RocReport().add_condition_auc_not_less_than(),
        CalibrationMetric(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than(),
        ClassPerformanceImbalance().add_condition_ratio_difference_not_greater_than(),
        NewLabelTrainTest().add_condition_new_labels_not_greater_than()
    )


def regression_suite() -> Suite:
    """Create 'Regression Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the regression model.
    """
    return Suite(
        'Regression Suite',
        RegressionSystematicError().add_condition_systematic_error_ratio_to_rmse_not_greater_than(),
        RegressionErrorDistribution().add_condition_kurtosis_not_less_than()
    )


def generic_performance_suite() -> Suite:
    """Create 'Generic Performance Suite'.

    The suite runs a set of checks that are meant to measure and detect performance abnormality in any model type.
    """
    return Suite(
        'Generic Performance Suite',
        PerformanceReport(),
        SimpleModelComparison().add_condition_ratio_not_less_than()
    )


def performance_suite() -> Suite:
    """Create 'Performance Suite'.

    The suite runs all checks that are meant to measure and detect performance abnormality in a model.

    The suite includes checks from 'Generic Performance Suite', 'Classification Suite'
    and 'Regression Suite'.
    """
    return Suite(
        'Performance Suite',
        generic_performance_suite(),
        classification_suite(),
        regression_suite()
    )
