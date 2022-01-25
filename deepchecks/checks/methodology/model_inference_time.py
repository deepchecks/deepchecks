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

from deepchecks.base.check_context import CheckRunContext
from deepchecks import SingleDatasetBaseCheck, CheckResult, ConditionResult
from deepchecks.utils.strings import format_number
from deepchecks.errors import DeepchecksValueError


__all__ = ['ModelInferenceTime']


MI = t.TypeVar('MI', bound='ModelInferenceTime')


class ModelInferenceTime(SingleDatasetBaseCheck):
    """Measure model average inference time (in seconds) per sample.

    Parameters
    ----------
    number_of_samples : int , default: 1000
        number of samples to use for inference, but if actual
        dataset is smaller then all samples will be used
    """

    def __init__(self, number_of_samples: int = 1000):
        self.number_of_samples = number_of_samples
        if number_of_samples == 0 or number_of_samples < 0:
            raise DeepchecksValueError('number_of_samples cannot be le than 0!')
        super().__init__()

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is of the type 'float' .

        Raises
        ------
        DeepchecksValueError
            If the test dataset is not a 'Dataset' instance with a label or
            if 'model' is not a scikit-learn-compatible fitted estimator instance.
        """
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        features = context.features
        model = context.model
        df = dataset.data[features]

        prediction_method = model.predict  # type: ignore

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

        Parameters
        ----------
        value : float , default: 0.001
            condition threshold.
        Returns
        -------
        MI
        """
        def condition(avarage_time: float) -> ConditionResult:
            if avarage_time >= value:
                return ConditionResult(False, details=(
                    'Found average inference time (in seconds) above threshold: '
                    f'{format_number(avarage_time, floating_point=8)}'
                ))
            else:
                return ConditionResult(True)

        return self.add_condition(condition_func=condition, name=(
            'Average model inference time for one sample is not '
            f'greater than {format_number(value, floating_point=8)}'
        ))
