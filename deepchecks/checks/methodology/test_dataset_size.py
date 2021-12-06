import typing as t
from deepchecks import Dataset, SingleDatasetBaseCheck, CheckResult, ConditionResult


__all__ = ['TestDatasetSize',]


TTestSize = t.TypeVar('TTestSize', bound='TestDatasetSize')


class TestDatasetSize(SingleDatasetBaseCheck):
    """Verify Test dataset size."""

    def run(self, train: Dataset, test: Dataset, model: object) -> CheckResult:
        """Run check instance.

        Args:
            train (Dataset): train dataset
            test (Dataset): test dataset
            model (object): a scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: with value of type Dict[str, int].
                Value contains two keys, 'train' - size of the train dataset
                and 'test' - size of the test dataset.

        Raises:
            DeepchecksValueError:
                if not dataset instances were provided;
                if datasets are empty;
                if datasets do not have features columns;
                if datasets do not have label column;
                if datasets do not share same features;
        """
        check_name = type(self).__name__
        Dataset.validate_dataset(train, check_name)
        Dataset.validate_dataset(test, check_name)
        train.validate_features(check_name)
        train.validate_label(check_name)
        test.validate_features(check_name)
        test.validate_label(check_name)
        train.validate_shared_features(test, check_name)
        train.validate_shared_label(test, check_name)
        return CheckResult(
            value={'train': train.n_samples, 'test': test.n_samples},
            header='Test dataset size.'
        )

    def add_condition_test_size_not_smaller_than(self: TTestSize, value: int = 100) -> TTestSize:
        """Add condition verifying that size of the test dataset is not smaller than X.

        Args:
            value (int): minimal allowed test dataset size.

        Returns:
            Self: current instance of the TestSize check.
        """
        def condition(check_result: t.Dict[str, int]) -> ConditionResult:
            return (
                ConditionResult(False, f'Test dataset is smaller than {value}.')
                if check_result['test'] <= value
                else ConditionResult(True)
            )

        return self.add_condition(
            name=f'Test dataset size is not smaller than {value}.',
            condition_func=condition
        )

    def add_condition_test_train_size_ratio_not_smaller_than(self: TTestSize, value: float = 0.1) -> TTestSize:
        """Add condition verifying that test-train size ratio is not smaller than X.

        Args:
            value (float): minimal allowed test-train ratio.

        Returns:
            Self: current instance of the TestSize check.
        """

        def condition(check_result: t.Dict[str, int]) -> ConditionResult:
            if check_result['train'] < check_result['test']:
                return ConditionResult(False, 'Train dataset is smaller than test dataset.')

            if (check_result['test'] / check_result['train']) <= value:
                return ConditionResult(False, f'Test-Train size ratio is smaller than {value}.')

            return ConditionResult(True)

        return self.add_condition(
            name=f'Test-Train size ratio is not smaller than {value}.',
            condition_func=condition
        )
