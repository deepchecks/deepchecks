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
from .check_json import CheckFailureJson, CheckResultJson
from .check_result import CheckFailure, CheckResult
from .checks import BaseCheck, DatasetKind, ModelOnlyBaseCheck, SingleDatasetBaseCheck, TrainTestBaseCheck
from .condition import Condition, ConditionCategory, ConditionResult
from .suite import BaseSuite, SuiteResult

__all__ = [
    'BaseCheck',
    'CheckResult',
    'CheckFailure',
    'CheckFailureJson',
    'CheckResultJson',
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
