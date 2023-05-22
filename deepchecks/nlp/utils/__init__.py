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
"""Utils package for nlp functionality."""

from deepchecks.nlp.utils.llm_utils import call_open_ai_completion_api
from deepchecks.nlp.utils.text_embeddings import calculate_builtin_embeddings
from deepchecks.nlp.utils.text_properties import calculate_builtin_properties

__all__ = [
    'calculate_builtin_properties',
    'calculate_builtin_embeddings',
    'call_open_ai_completion_api'
]
