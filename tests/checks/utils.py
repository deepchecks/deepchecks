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
