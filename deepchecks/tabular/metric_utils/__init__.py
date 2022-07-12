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
"""Module containing metrics utils."""

from .scorers import (DEFAULT_BINARY_SCORERS, DEFAULT_MULTICLASS_SCORERS, DEFAULT_REGRESSION_SCORERS,
                      DEFAULT_SCORERS_DICT, MULTICLASS_SCORERS_NON_AVERAGE, DeepcheckScorer, TaskType,
                      get_default_scorers, init_validate_scorers, task_type_check)

__all__ = [
    'TaskType',
    'task_type_check',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_REGRESSION_SCORERS',
    'DEFAULT_BINARY_SCORERS',
    'DEFAULT_MULTICLASS_SCORERS',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'DeepcheckScorer',
    'init_validate_scorers',
    'get_default_scorers'
]
