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

import pytest

from deepchecks.tabular.checks import (DominantFrequencyChange, ModelErrorAnalysis, SegmentPerformance,
                                       TrainTestFeatureDrift, TrainTestLabelDrift, TrainTestPredictionDrift,
                                       WholeDatasetDrift)


def test_deprecation_warning_label_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning, match='max_num_categories'):
        _ = TrainTestLabelDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = TrainTestLabelDrift()


def test_deprecation_warning_prediction_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning, match='max_num_categories'):
        _ = TrainTestPredictionDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = TrainTestPredictionDrift()


def test_deprecation_warning_feature_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning, match='max_num_categories'):
        _ = TrainTestFeatureDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = TrainTestFeatureDrift()


def test_deprecation_warning_whole_dataset_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning, match='max_num_categories'):
        _ = WholeDatasetDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = WholeDatasetDrift()


def test_deprecation_dominant_freq_change_warning():
    with pytest.warns(DeprecationWarning, match='The DominantFrequencyChange check is deprecated'):
        _ = DominantFrequencyChange()


def test_deprecation_model_error_analysis_warning():
    with pytest.warns(DeprecationWarning, match='The ModelErrorAnalysis check is deprecated and will be removed in the '
                                                '0.11 version. Please use the WeakSegmentsPerformance check instead.'):
        _ = ModelErrorAnalysis()


def test_deprecation_segment_performance_warning():
    with pytest.warns(DeprecationWarning, match='The SegmentPerformance check is deprecated and will be removed in the '
                                                '0.11 version. Please use the WeakSegmentsPerformance check instead.'):
        _ = SegmentPerformance()