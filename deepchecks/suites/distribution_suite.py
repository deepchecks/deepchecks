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
"""The predefined Data Distribution suite module."""
from deepchecks import Suite
from deepchecks.checks.distribution import TrainTestFeatureDrift, WholeDatasetDrift


def data_distribution_suite() -> Suite:
    """Create 'Data Distribution Suite'.

    The suite runs a check comparing the distributions of the training and test datasets.
    """
    return Suite(
        'Data Distribution',
        TrainTestFeatureDrift().add_condition_drift_score_not_greater_than(),
        WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than()
    )
