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
"""Utils functions for testing."""
import re
from typing import Union, Pattern
from hamcrest import all_of, has_property, matches_regexp
from hamcrest.core.matcher import Matcher
from deepchecks import ConditionCategory


__all__ = ['ANY_FLOAT_REGEXP', 'equal_condition_result']


ANY_FLOAT_REGEXP = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')
SCIENTIFIC_NOTATION_REGEXP = re.compile(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')


def equal_condition_result(
    is_pass: bool,
    name: str,
    details: Union[str, Pattern] = '',
    category: ConditionCategory = ConditionCategory.FAIL
) -> Matcher[Matcher[object]]:
    # Check if details is a regex class
    if hasattr(details, 'match'):
        details_matcher = matches_regexp(details)
    else:
        details_matcher = details

    return all_of(
        has_property('is_pass', is_pass),
        has_property('category', category),
        has_property('details', details_matcher),
        has_property('name', name)
    )
