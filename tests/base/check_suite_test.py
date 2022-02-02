# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""suites tests"""
import random
from hamcrest import assert_that, calling, raises, equal_to, is_

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Suite, SingleDatasetCheck, TrainTestCheck
from deepchecks.tabular import checks as tabular_checks



class SimpleDatasetCheck(SingleDatasetCheck):
    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        return CheckResult("Simple Check")


class SimpleTwoDatasetsCheck(TrainTestCheck):
    def run_logic(self, context) -> CheckResult:
        return CheckResult("Simple Check")


def test_suite_instantiation_with_incorrect_args():
    incorrect_check_suite_args = ("test suite", SimpleDatasetCheck(), object())
    assert_that(
        calling(Suite).with_args(*incorrect_check_suite_args),
        raises(DeepchecksValueError)
    )


def test_run_suite_with_incorrect_args():
    suite = Suite("test suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())

    # incorrect, at least one dataset (or model) must be provided
    args = {"train_dataset": None, "test_dataset": None,}
    assert_that(
        calling(suite.run).with_args(**args),
        raises(DeepchecksValueError, r'At least one dataset \(or model\) must be passed to the method!')
    )


def test_add_check_to_the_suite():
    number_of_checks = random.randint(0, 50)
    produce_checks = lambda count: [SimpleDatasetCheck() for _ in range(count)]

    first_suite = Suite("first suite", )
    second_suite = Suite("second suite", )

    assert_that(len(first_suite.checks), equal_to(0))
    assert_that(len(second_suite.checks), equal_to(0))

    for check in produce_checks(number_of_checks):
        first_suite.add(check)

    assert_that(len(first_suite.checks), equal_to(number_of_checks))

    second_suite.add(first_suite)
    assert_that(len(second_suite.checks), equal_to(number_of_checks))



def test_try_add_not_a_check_to_the_suite():
    suite = Suite("second suite")
    assert_that(
        calling(suite.add).with_args(object()),
        raises(DeepchecksValueError, 'Suite received unsupported object type: object')
    )


def test_try_add_check_suite_to_itself():
    suite = Suite("second suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    suite.add(suite)
    assert_that(len(suite.checks), equal_to(2))


def test_suite_static_indexes():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = Suite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)
    assert_that(len(suite.checks), equal_to(1))
    assert_that(suite[1], is_(second_check))


def test_access_removed_check_by_index():
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = Suite("first suite", first_check, second_check)

    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))

    suite.remove(0)

    assert_that(
        calling(suite.__getitem__).with_args(0),
        raises(DeepchecksValueError, 'No index 0 in suite')
    )


def test_try_remove_unexisting_check_from_the_suite():
    suite = Suite("first suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    assert_that(
        calling(suite.remove).with_args(3),
        raises(DeepchecksValueError, 'No index 3 in suite')
    )


def test_check_suite_instantiation_by_extending_another_check_suite():
    suite = Suite(
        "outer",
        tabular_checks.IsSingleValue(),
        Suite(
            "inner1",
            tabular_checks.MixedNulls(),
            Suite("inner2", tabular_checks.MixedDataTypes()),
            tabular_checks.PerformanceReport()
        )
    )

    assert all(not isinstance(c, Suite) for c in suite.checks)

    # assert that order of checks instances are preserved

    checks_types = [type(c) for c in suite.checks.values()]

    assert checks_types == [
        tabular_checks.IsSingleValue,
        tabular_checks.MixedNulls,
        tabular_checks.MixedDataTypes,
        tabular_checks.PerformanceReport
    ]
