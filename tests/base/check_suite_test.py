"""suites tests"""
import re
import pandas as pd
import numpy as np
from hamcrest import assert_that, calling, raises, equal_to, is_

from deepchecks import base
from deepchecks import checks as builtin_checks
from deepchecks.utils import DeepchecksValueError


class SimpleDatasetCheck(base.SingleDatasetBaseCheck):
    def run(self, dataset: base.Dataset, model: object = None) -> base.CheckResult:
        return base.CheckResult("Simple Check")


class SimpleTwoDatasetsCheck(base.CompareDatasetsBaseCheck):
    def run(self, first: base.Dataset, second: base.Dataset, model: object = None) -> base.CheckResult:
        return base.CheckResult("Simple Check")


def test_check_suite_instantiation_with_incorrect_args():
    incorrect_check_suite_args = ("test suite", SimpleDatasetCheck(), object())
    assert_that(
        calling(base.CheckSuite).with_args(*incorrect_check_suite_args),
        raises(DeepchecksValueError)
    )


def test_run_check_suite_with_incorrect_args(diabetes):
    train_dataset, test_dataset = diabetes
    suite = base.CheckSuite("test suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    
    # incorrect at least one dataset (or model) must be provided
    args = {"train_dataset": None, "test_dataset": None,} 
    assert_that(
        calling(suite.run).with_args(**args), 
        raises(ValueError, r'At least one dataset \(or model\) must be passed to the method!')
    )

    args = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "check_datasets_policy": "ttest" # incorrect policy literal
    }

    assert_that(
        calling(suite.run).with_args(**args), 
        raises(ValueError, r'check_datasets_policy must be one of \["both", "train", "test"\]')
    )


def test_try_add_not_a_check_to_the_suite():
    first_suite = base.CheckSuite("first suite", SimpleDatasetCheck())
    second_suite = base.CheckSuite("second suite")

    # should not raise an error
    second_suite.add(first_suite)
    second_suite.add(SimpleTwoDatasetsCheck())
    
    assert_that(
        calling(second_suite.add).with_args(object()),
        raises(DeepchecksValueError, 'CheckSuite receives only `BaseCheck` objects but got: object')
    )


def test_try_add_check_suite_to_itself():
    first_suite = base.CheckSuite("first suite", SimpleDatasetCheck())
    second_suite = base.CheckSuite("second suite", first_suite, SimpleTwoDatasetsCheck())

    assert_that(len(second_suite.checks), equal_to(2))
    second_suite.add(second_suite)
    assert_that(len(second_suite.checks), equal_to(2))


def test_suite_static_indexes():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = base.CheckSuite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)
    assert_that(len(suite.checks), equal_to(1))
    assert_that(suite[1], is_(second_check))


def test_access_removed_check_by_index():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = base.CheckSuite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)

    assert_that(
        calling(suite.__getitem__).with_args(0),
        raises(DeepchecksValueError, 'No index 0 in suite')
    )


def test_try_remove_unexisting_check_from_the_suite():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = base.CheckSuite("first suite", first_check, second_check)
    
    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    assert_that(
        calling(suite.remove).with_args(3),
        raises(DeepchecksValueError, 'No index 3 in suite')
    )


def test_check_suite_instantiation_by_extending_another_check_suite():
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
