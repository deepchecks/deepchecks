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
"""Contains unit tests for the tabular package deprecation warnings."""
import warnings

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SegmentPerformance, WholeDatasetDrift, SimpleModelComparison, \
    MultiModelPerformanceReport, RegressionSystematicError


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
