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
"""Module for base classes.

Import objects to be available in parent deepchecks module.
"""
from .suite import BaseSuite, SuiteResult
from .condition import Condition, ConditionResult, ConditionCategory
from .check import (
    BaseCheck,
    CheckResult,
    CheckFailure,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
    DatasetKind
)


__all__ = [
    'BaseCheck',
    'CheckResult',
    'CheckFailure',
    'Condition',
    'ConditionResult',
    'ConditionCategory',
    'BaseSuite',
    'SuiteResult',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'DatasetKind'
]
