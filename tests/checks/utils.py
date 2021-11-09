from hamcrest import all_of, has_property, matches_regexp

from mlchecks import ConditionCategory


def equal_condition_result(is_pass: bool, name: str, details: str = '',
                           category: ConditionCategory = ConditionCategory.FAIL):
    return all_of(
        has_property('is_pass', is_pass),
        has_property('category', category),
        has_property('details', matches_regexp(details)),
        has_property('name', name)
    )
