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

from deepchecks.nlp.utils.llm_utils import calculate_embeddings_for_text, call_open_ai_completion_api

__all__ = [
    'calculate_embeddings_for_text',
    'call_open_ai_completion_api'
]
