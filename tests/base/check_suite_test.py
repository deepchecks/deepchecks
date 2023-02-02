# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
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
from typing import List

from hamcrest import all_of, assert_that, calling, equal_to, has_entry, has_items, has_length, instance_of, is_, raises

from deepchecks import __version__
from deepchecks.core import CheckFailure, CheckResult, ConditionCategory, ConditionResult, SuiteResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.suite import BaseSuite
from deepchecks.tabular import SingleDatasetCheck, Suite, TrainTestCheck
from deepchecks.tabular import checks as tabular_checks
from deepchecks.tabular.suites import model_evaluation


class SimpleDatasetCheck(SingleDatasetCheck):
    def run_logic(self, context, dataset_kind) -> CheckResult:
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
    args = {"train_dataset": None, "test_dataset": None, }
    assert_that(
        calling(suite.run).with_args(**args),
        raises(DeepchecksValueError, r"At least one dataset \(or model\) must be passed to the method!")
    )


def test_add_check_to_the_suite():
    number_of_checks = random.randint(0, 50)
    def produce_checks(count): return [SimpleDatasetCheck() for _ in range(count)]

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
        raises(DeepchecksValueError, "Suite received unsupported object type: object")
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
        raises(DeepchecksValueError, "No index 0 in suite")
    )


def test_try_remove_unexisting_check_from_the_suite():
    suite = Suite("first suite", SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    assert_that(
        calling(suite.remove).with_args(3),
        raises(DeepchecksValueError, "No index 3 in suite")
    )


def test_check_suite_instantiation_by_extending_another_check_suite():
    suite = Suite(
        "outer",
        tabular_checks.IsSingleValue(),
        Suite(
            "inner1",
            tabular_checks.MixedNulls(),
            Suite("inner2", tabular_checks.MixedDataTypes()),
            tabular_checks.TrainTestPerformance()
        )
    )

    assert all(not isinstance(c, Suite) for c in suite.checks)

    # assert that order of checks instances are preserved

    checks_types = [type(c) for c in suite.checks.values()]

    assert checks_types == [
        tabular_checks.IsSingleValue,
        tabular_checks.MixedNulls,
        tabular_checks.MixedDataTypes,
        tabular_checks.TrainTestPerformance
    ]


def test_suite_result_checks_not_passed():
    # Arrange
    result1 = CheckResult(0, 'check1')
    result1.conditions_results = [ConditionResult(ConditionCategory.PASS)]
    result2 = CheckResult(0, 'check2')
    result2.conditions_results = [ConditionResult(ConditionCategory.WARN)]
    result3 = CheckResult(0, 'check3')
    result3.conditions_results = [ConditionResult(ConditionCategory.FAIL)]

    # Act & Assert
    not_passed_checks = SuiteResult('test', [result1, result2]).get_not_passed_checks()
    assert_that(not_passed_checks, has_length(1))
    not_passed_checks = SuiteResult('test', [result1, result2]).get_not_passed_checks(fail_if_warning=False)
    assert_that(not_passed_checks, has_length(0))
    not_passed_checks = SuiteResult('test', [result1, result2, result3]).get_not_passed_checks()
    assert_that(not_passed_checks, has_length(2))


def test_suite_result_passed_fn():
    # Arrange
    result1 = CheckResult(0, 'check1')
    result1.conditions_results = [ConditionResult(ConditionCategory.PASS)]
    result2 = CheckResult(0, 'check2')
    result2.conditions_results = [ConditionResult(ConditionCategory.WARN)]
    result3 = CheckResult(0, 'check3')
    result3.conditions_results = [ConditionResult(ConditionCategory.FAIL)]
    result4 = CheckFailure(tabular_checks.IsSingleValue(), DeepchecksValueError(''))

    # Act & Assert
    passed = SuiteResult('test', [result1, result2]).passed()
    assert_that(passed, equal_to(False))
    passed = SuiteResult('test', [result1, result2]).passed(fail_if_warning=False)
    assert_that(passed, equal_to(True))
    passed = SuiteResult('test', [result1, result2, result3]).passed(fail_if_warning=False)
    assert_that(passed, equal_to(False))
    passed = SuiteResult('test', [result1, result4]).passed()
    assert_that(passed, equal_to(True))
    passed = SuiteResult('test', [result1, result4]).passed(fail_if_check_not_run=True)
    assert_that(passed, equal_to(False))


def test_config():
    model_eval_suite = model_evaluation()
    check_amount = len(model_eval_suite.checks)

    suite_mod = model_eval_suite.config()

    assert_that(suite_mod, all_of(
        has_entry('module_name', 'deepchecks.tabular.suite'),
        has_entry('class_name', 'Suite'),
        has_entry('name', 'Model Evaluation Suite'),
        has_entry('version', __version__),
        has_entry('checks', instance_of(list))
    ))

    conf_suite_mod = BaseSuite.from_config(suite_mod)
    assert_that(conf_suite_mod.name, equal_to('Model Evaluation Suite'))
    assert_that(conf_suite_mod.checks.values(), has_length(check_amount))
