"""
    suites tests
"""
from deepchecks import base
from deepchecks import checks as builtin_checks


def test_check_suite():
    suite = base.CheckSuite(
        "outer",
        builtin_checks.IsSingleValue(),
        base.CheckSuite(
            "inner1",
            builtin_checks.MixedNulls(),
            base.CheckSuite("inner2", builtin_checks.MixedTypes()),
            builtin_checks.TrainTestDifferenceOverfit()
        )
    )

    assert all(not isinstance(c, base.CheckSuite) for c in suite.checks)

    # assert that order of checks instances are preserved

    checks_types = [type(c) for c in suite.checks.values()]

    assert checks_types == [
        builtin_checks.IsSingleValue,
        builtin_checks.MixedNulls,
        builtin_checks.MixedTypes,
        builtin_checks.TrainTestDifferenceOverfit
    ]
