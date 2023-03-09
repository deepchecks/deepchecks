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
"""The task type module containing the TaskType enum."""
from enum import Enum

__all__ = ['TaskType']


class TaskType(Enum):
    """Enum containing supported task types."""

    TEXT_CLASSIFICATION = 'text_classification'
    TOKEN_CLASSIFICATION = 'token_classification'
    OTHER = 'other'
