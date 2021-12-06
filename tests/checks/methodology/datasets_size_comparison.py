import pandas as pd
from hamcrest import assert_that, calling, raises, has_entries, instance_of, all_of

from deepchecks import CheckResult, ConditionCategory
from deepchecks.checks import DatasetsSizeComparison
from deepchecks.errors import DeepchecksValueError

from tests.checks.utils import equal_condition_result


def test_test_dataset_size_check(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model

    check_result = DatasetsSizeComparison().run(train, test, model)

    assert_that(check_result, instance_of(CheckResult))
    assert_that(check_result.value, instance_of(pd.DataFrame))
    assert_that(check_result.value, has_entries({
        "train": all_of(
            instance_of(pd.Series),
            has_entries({"size": instance_of(int)})
        ),
        "test": all_of(
            instance_of(pd.Series),
            has_entries({"size": instance_of(int)})
        )
    }))


def test_run_test_dataset_size_check_with_empty_datasets(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model

    train._data = train._data[0:0]
    test._data = test._data[0:0]

    assert_that(
        calling(DatasetsSizeComparison().run).with_args(train, test, model),
        raises(DeepchecksValueError, 'Check DatasetsSizeComparison required a non-empty dataset')
    )


def test_test_dataset_size_check_with_condition_that_should_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_size_not_smaller_than(10)

    check_result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(actual=condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=True,
        name='Test dataset size is not smaller than 10.',
        details='',
        category=ConditionCategory.FAIL
    ))


def test_test_dataset_size_check_with_condition_that_should_not_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_size_not_smaller_than(10_000)

    check_result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(actual=condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=False,
        name='Test dataset size is not smaller than 10000.',
        details=r'Test dataset is smaller than 10000.',
        category=ConditionCategory.FAIL
    ))


def test_test_dataset_size_check_with_size_ratio_condition_that_should_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_train_size_ratio_not_smaller_than(0.2)

    check_result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(actual=condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=True,
        name='Test-Train size ratio is not smaller than 0.2.',
        details='',
        category=ConditionCategory.FAIL
    ))


def test_test_dataset_size_check_with_size_ratio_condition_that_should_not_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_train_size_ratio_not_smaller_than(0.8)

    check_result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(actual=condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=False,
        name='Test-Train size ratio is not smaller than 0.8.',
        details=r'Test-Train size ratio is smaller than 0.8.',
        category=ConditionCategory.FAIL
    ))
