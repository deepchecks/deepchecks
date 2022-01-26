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
"""Package for tabular functionality."""
from .dataset import Dataset
from .base import (
    TabularContext,
    TabularCheck,
    Suite,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
    ModelComparisonContext,
    ModelComparisonCheck,
    ModelComparisonSuite
)


__all__ = [
    "Dataset",
    "TabularContext",
    "TabularCheck",
    "SingleDatasetBaseCheck",
    "TrainTestBaseCheck",
    "ModelOnlyBaseCheck",
    "Suite",
    "ModelComparisonContext",
    "ModelComparisonCheck",
    "ModelComparisonSuite",
]
