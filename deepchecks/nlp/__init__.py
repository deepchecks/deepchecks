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
"""Package for nlp functionality."""
from .base_checks import SingleDatasetCheck, TrainTestCheck
from .context import Context
from .suite import Suite
from .text_data import TextData

__all__ = [
    'TextData',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'Suite',
    'Context'
]
