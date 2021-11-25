"""The model inference time check module."""
import typing as t
import timeit

from deepchecks import SingleDatasetBaseCheck, CheckResult, Dataset, ConditionResult
from deepchecks.utils import model_type_validation, DeepchecksValueError
from deepchecks.string_utils import format_number


__all__ = ['ModelInferenceTimeCheck']


MI = t.TypeVar('MI', bound='ModelInferenceTimeCheck')


class ModelInferenceTimeCheck(SingleDatasetBaseCheck):
    """Measure model average inference time (in seconds) per sample."""

    NUMBER_OF_SAMPLES: t.ClassVar[int] = 1000

    def run(self, dataset: Dataset, model: object) -> CheckResult:
        """Run check.

        Args:
            train_dataset: a dataset that was used to train the model
            test_dataset: a dataset to validate the model correctness
            model: a model to measure inference time

        Raises:
            DeepchecksValueError: If the 'test_dataset' is not a 'Dataset' instance with a label or
                if 'model' is not a scikit-learn-compatible fitted estimator instance

        """
        return self._model_inference_time_check(dataset, model)

    def _model_inference_time_check(
        self,
        dataset: Dataset,
        model: object # TODO: find more precise type for model
    ) -> CheckResult:
        check_name = type(self).__name__
        Dataset.validate_dataset(dataset, check_name)
        model_type_validation(model)

        if dataset.features_columns() is None:
            raise DeepchecksValueError(f'Check {check_name} requires dataset with the features!')

        prediction_method = model.predict # type: ignore
        df = dataset.features_columns()[:self.NUMBER_OF_SAMPLES] # type: ignore

        result = timeit.timeit(
            'predict(*args)',
            globals={'predict': prediction_method, 'args': (df.to_numpy(),)},
            number=1
        )

        result = result / self.NUMBER_OF_SAMPLES

        return CheckResult(value=result, check=type(self), display=(
            'Average model inference time of one sample (in seconds) '
            f'equal to {format_number(result, floating_point=8)}'
        ))

    def add_condition_inference_time_is_not_greater_than(self: MI, value: float = 0.001) -> MI:
        """Add condition checking that the average model inference time (in seconds)
        per sample is not greater than X

        Args:
            value: condition threshold
        """
        def condition(avarage_time: float) -> ConditionResult:
            if avarage_time >= value:
                return ConditionResult(False, details=(
                    'Average model inference time of one sample (in seconds) '
                    f'is greater than {format_number(value, floating_point=8)}'
                ))
            else:
                return ConditionResult(True)

        return self.add_condition(condition_func=condition, name=(
            'Average model inference time for one sample is not '
            f'greater than {format_number(value, floating_point=8)}'
        ))
