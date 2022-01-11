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
"""Module for base classes.

Import objects to be available in parent deepchecks module.
"""
from .dataset import Dataset
from .check import (
    BaseCheck,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
    CheckResult,
    CheckFailure,
    ModelComparisonBaseCheck,
    ModelComparisonContext
)
from .suite import (
    BaseSuite,
    Suite,
    SuiteResult,
    ModelComparisonSuite
)
from .condition import (
    Condition,
    ConditionResult,
    ConditionCategory
)


__all__ = [
    'Dataset',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ModelComparisonBaseCheck',
    'ModelComparisonContext',
    'CheckResult',
    'CheckFailure',
    'Condition',
    'ConditionResult',
    'ConditionCategory',
    'BaseSuite',
    'Suite',
    'SuiteResult',
    'ModelComparisonSuite'
]
