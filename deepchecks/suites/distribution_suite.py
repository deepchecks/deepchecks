# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""The predefined Data Distribution suite module."""
from deepchecks import Suite
from deepchecks.checks.distribution import TrainTestDrift
from deepchecks.checks.distribution import TrustScoreComparison


def data_distribution_suite() -> Suite:
    """Create 'Data Distribution Suite'.

    The suite runs a check comparing the distributions of the training and test datasets.
    """
    return Suite(
        'Data Distribution',
        TrainTestDrift().add_condition_drift_score_not_greater_than(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than()
    )
