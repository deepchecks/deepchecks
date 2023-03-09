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
"""Contains unit tests for the vision package deprecation warnings."""

import pytest

from deepchecks.vision.checks import TrainTestPredictionDrift, TrainTestLabelDrift


def test_deprecation_warning_train_test_label_drift():
    with pytest.warns(DeprecationWarning, match="The TrainTestLabelDrift check is deprecated and will be removed in "
                                                "the 0.14 version.Please use the LabelDrift check instead."):
        _ = TrainTestLabelDrift()


def test_deprecation_warning_train_test_prediction_drift():
    with pytest.warns(DeprecationWarning, match="The TrainTestPredictionDrift check is deprecated and will be removed "
                                                "in the 0.14 version.Please use the PredictionDrift check instead."):
        _ = TrainTestPredictionDrift()
