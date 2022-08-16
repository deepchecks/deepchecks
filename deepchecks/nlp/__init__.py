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
"""Package for nlp functionality."""
from .base_checks import SingleDatasetCheck, TrainTestCheck
from .suite import Suite

try:
    import datasets  # noqa: F401
    import seqeval  # noqa: F401
except ImportError as error:
    raise ImportError("datasets (HuggingFace) or seqeval are not installed. Install requirments specified"
                      " in nlp-requirements.txt in order to use deepchecks.nlp functionalities.") from error


__all__ = [
    "SingleDatasetCheck",
    "TrainTestCheck",
    "Suite",
]
