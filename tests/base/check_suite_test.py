"""suites tests"""
import random
from hamcrest import assert_that, calling, raises, equal_to, is_

from deepchecks import base
from deepchecks import checks as builtin_checks
from deepchecks.errors import DeepchecksValueError


class SimpleDatasetCheck(base.SingleDatasetBaseCheck):
    def run(self, dataset: base.Dataset, model: object = None) -> base.CheckResult:
        return base.CheckResult("Simple Check")


class SimpleTwoDatasetsCheck(base.CompareDatasetsBaseCheck):
    def run(self, first: base.Dataset, second: base.Dataset, model: object = None) -> base.CheckResult:
        return base.CheckResult("Simple Check")


def test_check_suite_instantiation_with_incorrect_args():
    incorrect_check_suite_args = ("test suite", SimpleDatasetCheck(), object())
    assert_that(
        calling(base.Suite).with_args(*incorrect_check_suite_args),
        raises(DeepchecksValueError)
    )


def test_run_check_suite_with_incorrect_args(diabetes):
    train_dataset, test_dataset = diabetes
    suite = base.Suite("test suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    
    # incorrect, at least one dataset (or model) must be provided
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


def test_add_check_to_the_suite():
    number_of_checks = random.randint(0, 50)
    produce_checks = lambda count: [SimpleDatasetCheck() for _ in range(count)]

    first_suite = base.Suite("first suite", )
    second_suite = base.Suite("second suite", )
    
    assert_that(len(first_suite.checks), equal_to(0))
    assert_that(len(second_suite.checks), equal_to(0))
    
    for check in produce_checks(number_of_checks):
        first_suite.add(check)
    
    assert_that(len(first_suite.checks), equal_to(number_of_checks))

    second_suite.add(first_suite)
    assert_that(len(second_suite.checks), equal_to(number_of_checks))
    


def test_try_add_not_a_check_to_the_suite():
    suite = base.Suite("second suite")
    assert_that(
        calling(suite.add).with_args(object()),
        raises(DeepchecksValueError, 'Suite receives only `BaseCheck` objects but got: object')
    )


def test_try_add_check_suite_to_itself():
    suite = base.Suite("second suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    suite.add(suite)
    assert_that(len(suite.checks), equal_to(2))


def test_suite_static_indexes():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = base.Suite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)
    assert_that(len(suite.checks), equal_to(1))
    assert_that(suite[1], is_(second_check))


def test_access_removed_check_by_index():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = base.Suite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)

    assert_that(
        calling(suite.__getitem__).with_args(0),
        raises(DeepchecksValueError, 'No index 0 in suite')
    )


def test_try_remove_unexisting_check_from_the_suite():
    suite = base.Suite("first suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    assert_that(
        calling(suite.remove).with_args(3),
        raises(DeepchecksValueError, 'No index 3 in suite')
    )


def test_check_suite_instantiation_by_extending_another_check_suite():
    suite = base.Suite(
        "outer",
        builtin_checks.IsSingleValue(),
        base.Suite(
            "inner1",
            builtin_checks.MixedNulls(),
            base.Suite("inner2", builtin_checks.MixedTypes()),
            builtin_checks.TrainTestDifferenceOverfit()
        )
    )

    assert all(not isinstance(c, base.Suite) for c in suite.checks)

    # assert that order of checks instances are preserved

    checks_types = [type(c) for c in suite.checks.values()]

    assert checks_types == [
        builtin_checks.IsSingleValue,
        builtin_checks.MixedNulls,
        builtin_checks.MixedTypes,
        builtin_checks.TrainTestDifferenceOverfit
    ]
