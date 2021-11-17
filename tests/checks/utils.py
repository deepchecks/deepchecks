"""Utils functions for testing."""
import re
from typing import Union, Pattern
from hamcrest import all_of, has_property, matches_regexp
from hamcrest.core.matcher import Matcher
from deepchecks import ConditionCategory


__all__ = ['ANY_FLOAT_REGEXP', 'equal_condition_result']


ANY_FLOAT_REGEXP = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')


def equal_condition_result(
    is_pass: bool,
    name: str,
    details: Union[str, Pattern] = '',
    category: ConditionCategory = ConditionCategory.FAIL
) -> Matcher[Matcher[object]]:
    # Check if details is a regex class
    if 'match' in dir(details):
        details_matcher = matches_regexp(details)
    else:
        details_matcher = details

    return all_of(
        has_property('is_pass', is_pass),
        has_property('category', category),
        has_property('details', details_matcher),
        has_property('name', name)
    )
