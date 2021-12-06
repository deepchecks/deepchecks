# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Module contains all data integrity checks."""
from .mixed_nulls import *
from .string_mismatch import *
from .mixed_types import *
from .is_single_value import *
from .special_chars import *
from .string_length_out_of_bounds import *
from .string_mismatch_comparison import *
from .rare_format_detection import *
from .dominant_frequency_change import *
from .data_duplicates import *
from .new_category import *
from .new_label import *
from .label_ambiguity import *
