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

from deepchecks.tabular.checks import TrainTestLabelDrift, TrainTestPredictionDrift, TrainTestFeatureDrift, \
    WholeDatasetDrift


def test_deprecation_warning_label_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning):
        check = TrainTestLabelDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = TrainTestLabelDrift()


def test_deprecation_warning_prediction_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning):
        check = TrainTestPredictionDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = TrainTestPredictionDrift()


def test_deprecation_warning_feature_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning):
        check = TrainTestFeatureDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = TrainTestFeatureDrift()


def test_deprecation_warning_whole_dataset_drift():
    # Test that warning is raised when max_num_categories has value:
    with pytest.warns(DeprecationWarning):
        check = WholeDatasetDrift(max_num_categories=10)

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = WholeDatasetDrift()