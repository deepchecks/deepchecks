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
"""Top module for deepchecks library."""
import matplotlib
from .utils.ipython import is_notebook
from .base import (
    Dataset,
    BaseCheck,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
    ModelComparisonBaseCheck,
    CheckResult,
    CheckFailure,
    Condition,
    ConditionResult,
    ConditionCategory,
    BaseSuite,
    Suite,
    SuiteResult,
    ModelComparisonSuite,
)


__all__ = [
    'Dataset',
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ModelComparisonBaseCheck',
    'CheckResult',
    'CheckFailure',
    'Condition',
    'ConditionResult',
    'ConditionCategory',
    'BaseSuite',
    'Suite',
    'SuiteResult',
    'ModelComparisonSuite',
]


# Matplotlib has multiple backends. If we are in a context that does not support GUI (For example, during unit tests)
# we can't use a GUI backend. Thus we must use a non-GUI backend.
if not is_notebook():
    matplotlib.use('Agg')
