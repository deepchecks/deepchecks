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
"""Module contains all data integrity checks."""
from .mixed_nulls import *
from .string_mismatch import *
from .mixed_types import *
from .is_single_value import *
from .special_chars import *
from .string_length_out_of_bounds import *
from .string_mismatch_comparison import *
from .dominant_frequency_change import *
from .data_duplicates import *
from .new_category import *
from .new_label import *
from .label_ambiguity import *
