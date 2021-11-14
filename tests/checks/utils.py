"""Utils functions for testing."""
from re import Pattern
from typing import Union, Pattern as PatternType

from hamcrest import all_of, has_property, matches_regexp

from mlchecks import ConditionCategory


def equal_condition_result(is_pass: bool, name: str, details: Union[str, PatternType] = '',
                           category: ConditionCategory = ConditionCategory.FAIL):
    if isinstance(details, Pattern):
        details_matcher = matches_regexp(details)
    else:
        details_matcher = details

    return all_of(
        has_property('is_pass', is_pass),
        has_property('category', category),
        has_property('details', details_matcher),
        has_property('name', name)
    )
