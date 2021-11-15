"""Utils functions for testing."""
from typing import Union, Pattern

from hamcrest import all_of, has_property, matches_regexp

from mlchecks import ConditionCategory


def equal_condition_result(is_pass: bool, name: str, details: Union[str, Pattern] = '',
                           category: ConditionCategory = ConditionCategory.FAIL):
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
