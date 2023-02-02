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
"""Utils functions for testing."""
import re
from typing import Pattern, Union

from hamcrest import all_of, has_property, is_in, matches_regexp
from hamcrest.core.matcher import Matcher

from deepchecks.core import ConditionCategory

__all__ = ['ANY_FLOAT_REGEXP', 'equal_condition_result']


ANY_FLOAT_REGEXP = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')
SCIENTIFIC_NOTATION_REGEXP = re.compile(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')


def equal_condition_result(
    is_pass: bool,
    name: str,
    details: Union[str, Pattern] = '',
    category: ConditionCategory = None
) -> Matcher[Matcher[object]]:
    if category is None:
        possible_categories = [ConditionCategory.PASS] if is_pass else [ConditionCategory.FAIL, ConditionCategory.WARN]
    else:
        possible_categories = [category]

    # Check if details is a regex class
    if hasattr(details, 'match'):
        details_matcher = matches_regexp(details)
    else:
        details_matcher = details

    return all_of(
        has_property('category', is_in(possible_categories)),
        has_property('details', details_matcher),
        has_property('name', name)
    )
