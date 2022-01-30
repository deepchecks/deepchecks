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
"""Module containing metrics utils."""

from .metrics import (
    task_type_check,
    get_scorers_list,
    calculate_metrics
)

__all__ = [
    'task_type_check',
    'get_scorers_list',
    'calculate_metrics',
]
