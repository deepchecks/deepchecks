"""The model inference time check module."""
import typing as t
import timeit

from deepchecks import TrainTestBaseCheck, CheckResult, Dataset, ConditionResult
from deepchecks.utils import model_type_validation
from deepchecks.string_utils import format_number


__all__ = ['ModelInferenceTimeCheck']


MI = t.TypeVar('MI', bound='ModelInferenceTimeCheck')


class ModelInferenceTimeCheck(TrainTestBaseCheck):
    """Measure model average inference time (in seconds) per sample."""

    def run(
        self,
        train_dataset: t.Optional[Dataset],
        test_dataset: Dataset,
        model: object
    ) -> CheckResult:
        """Run check.

        Args:
            train_dataset: a dataset that was used to train the model
            test_dataset: a dataset to validate the model correctness
            model: a model to measure inference time

        Raises:
            DeepchecksValueError: If the 'test_dataset' is not a 'Dataset' instance with a label or
                if 'model' is not an instance of the 'BaseEstimator' or 'CatBoost'

        """
        return self._model_inference_time_check(test_dataset, model)

    def _model_inference_time_check(
        self,
        test_dataset: Dataset,
        model: object # TODO: find more precise type for model
    ) -> CheckResult:
        Dataset.validate_dataset(test_dataset, type(self).__name__)
        model_type_validation(model)

        timeing = []
        prediction_method = model.predict # type: ignore
        df = test_dataset.features_columns()

        assert df is not None, "Internal Error! 'dataset._features' var was not initialized!"  # pylint: disable=inconsistent-quotes

        for _, series in df.iterrows():
            features = series.array.reshape(1, -1) # type: ignore
            timeing.append(timeit.timeit(
                'predict(*args)',
                globals={'predict': prediction_method, 'args': (features,)},
                number=1
            ))

        result = sum(timeing) / len(timeing)

        return CheckResult(value=result, check=type(self), display=(
            'Average model inference time of one sample (in seconds) '
            f'equal to {format_number(result, floating_point=8)}'
        ))

    def add_condition_inference_time_is_not_greater_than(self: MI, value: float) -> MI:
        """Add Condition.

        Add condition that will check average model inference time (in seconds)
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
            'Average model inference time of one sample is not '
            f'greater than {format_number(value, floating_point=8)}'
        ))
