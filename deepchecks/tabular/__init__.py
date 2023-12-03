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
"""Package for tabular functionality."""
from deepchecks.tabular.base_checks import ModelComparisonCheck, ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.model_base import ModelComparisonContext, ModelComparisonSuite
from deepchecks.tabular.suite import Suite

__all__ = [
    "Dataset",
    "Context",
    "SingleDatasetCheck",
    "TrainTestCheck",
    "ModelOnlyCheck",
    "Suite",
    "ModelComparisonContext",
    "ModelComparisonCheck",
    "ModelComparisonSuite",
]
