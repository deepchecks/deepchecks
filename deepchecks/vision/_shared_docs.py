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
"""Module with common docstrings."""
from deepchecks.utils.decorators import Substitution

_shared_docs = {}

_shared_docs['additional_run_params'] = """
random_state : int
    A seed to set for pseudo-random functions
with_display : bool , default: True
    flag that determines if checks will calculate display (redundant in some checks).
""".strip('\n')

_shared_docs['additional_check_init_params'] = """
n_samples : Optional[int] , default : 10000
    Number of samples to use for the check. If None, all samples will be used.
""".strip('\n')

_shared_docs['property_aggregation_method_argument'] = """
argument for the reduce_output functionality, decides how to aggregate the individual properties drift scores
for a collective score between 0 and 1. Possible values are:
'mean': Mean of all properties scores.
'none': No averaging. Return a dict with a drift score for each property.
'max': Maximum of all the properties drift scores.
""".strip('\n')

docstrings = Substitution(**_shared_docs)
