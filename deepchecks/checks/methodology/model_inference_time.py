# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The model inference time check module."""
import typing as t
import timeit

import numpy as np
import pandas as pd

from deepchecks import SingleDatasetBaseCheck, CheckResult, Dataset, ConditionResult
from deepchecks.utils.validation import validate_model
from deepchecks.utils.strings import format_number
from deepchecks.errors import DeepchecksValueError


__all__ = ['ModelInferenceTimeCheck']


MI = t.TypeVar('MI', bound='ModelInferenceTimeCheck')


class ModelInferenceTimeCheck(SingleDatasetBaseCheck):
    """Measure model average inference time (in seconds) per sample.

    Args:
        number_of_samples (int):
            number of samples to use for inference, but if actual
            dataset is smaller then all samples will be used
    """

    def __init__(self, number_of_samples: int = 1000):
        self.number_of_samples = number_of_samples
        if number_of_samples == 0 or number_of_samples < 0:
            raise DeepchecksValueError('number_of_samples cannot be le than 0!')
        super().__init__()

    def run(self, dataset: Dataset, model: object) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): samples that will be used to measure inference time
            model (BaseEstimator): a model to measure inference time

        Returns:
            CheckResult:
                value is of the type 'float'

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
        Dataset.validate_dataset(dataset)
        Dataset.validate_features(dataset)
        validate_model(dataset, model)

        prediction_method = model.predict # type: ignore
        df = t.cast(pd.DataFrame, dataset.features_columns)

        number_of_samples = len(df) if len(df) < self.number_of_samples else self.number_of_samples
        df = df.sample(n=number_of_samples, random_state=np.random.randint(number_of_samples))

        result = timeit.timeit(
            'predict(*args)',
            globals={'predict': prediction_method, 'args': (df,)},
            number=1
        )

        result = result / number_of_samples

        return CheckResult(value=result, display=(
            'Average model inference time for one sample (in seconds): '
            f'{format_number(result, floating_point=8)}'
        ))

    def add_condition_inference_time_is_not_greater_than(self: MI, value: float = 0.001) -> MI:
        """Add condition - checking that the average model inference time (in seconds) per sample is not greater than X.

        Args:
            value: condition threshold
        """
        def condition(avarage_time: float) -> ConditionResult:
            if avarage_time >= value:
                return ConditionResult(False, details=(
                    'Average model inference time for one sample (in seconds) '
                    f'is {format_number(avarage_time, floating_point=8)}'
                ))
            else:
                return ConditionResult(True)

        return self.add_condition(condition_func=condition, name=(
            'Average model inference time for one sample is not '
            f'greater than {format_number(value, floating_point=8)}'
        ))
