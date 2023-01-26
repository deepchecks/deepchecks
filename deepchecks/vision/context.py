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
"""Module for base vision context."""
from typing import Optional

from deepchecks.core import CheckFailure, CheckResult, DatasetKind
from deepchecks.core.errors import DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.vision_data import TaskType, VisionData
from deepchecks.vision.vision_data.utils import set_seeds, validate_vision_data_compatibility

__all__ = ['Context']


@docstrings
class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train : Optional[VisionData] , default: None
        VisionData object, representing data the model was fitted on
    test : Optional[VisionData] , default: None
        VisionData object, representing data the models predicts on
    {additional_run_params:indent}
    """

    def __init__(
            self,
            train: Optional[VisionData] = None,
            test: Optional[VisionData] = None,
            random_state: int = 42,
            with_display: bool = True,
    ):
        # Validations
        if train is None and test is None:
            raise DeepchecksValueError('At least one dataset must be passed to the method!')
        if test is not None and train is None:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')

        train.init_cache()
        train.name = DEFAULT_DATASET_NAMES[0] if train.name is None else train.name
        self._task_type = train.task_type

        if test is not None:
            validate_vision_data_compatibility(train, test)
            test.init_cache()
            test.name = DEFAULT_DATASET_NAMES[1] if test.name is None else test.name

        self._train = train
        self._test = test
        self._with_display = with_display
        self.random_state = random_state
        set_seeds(random_state)

    @property
    def task_type(self) -> TaskType:
        """Return the common task type of the datasets."""
        return self._task_type

    @property
    def with_display(self) -> bool:
        """Return the with_display flag."""
        return self._with_display

    @property
    def train(self) -> VisionData:
        """Return train if exists, otherwise raise error."""
        if self._train is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self) -> VisionData:
        """Return test if exists, otherwise raise error."""
        if self._test is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without test dataset')
        return self._test

    def have_test(self):
        """Return whether there is test dataset defined."""
        return self._test is not None

    def assert_task_type(self, *expected_types: TaskType):
        """Assert task_type matching given types."""
        if self.train.task_type not in expected_types:
            raise DeepchecksNotSupportedError(
                f'Check is irrelevant for task of type {self.train.task_type}')
        return True

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant VisionData by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')

    def finalize_check_result(self, check_result, check, dataset_kind: DatasetKind = None):
        """Run final processing on a check result which includes validation and conditions processing."""
        # Validate the check result type
        if isinstance(check_result, CheckFailure):
            return
        if not isinstance(check_result, CheckResult):
            raise DeepchecksValueError(f'Check {check.name()} expected to return CheckResult but got: '
                                       + type(check_result).__name__)

        # Set reference between the check result and check
        check_result.check = check
        # Calculate conditions results
        check_result.process_conditions()

        # Add sampling footnote if needed
        if not hasattr(check, 'n_samples') or getattr(check, 'n_samples') is None:
            return

        n_samples = getattr(check, 'n_samples')
        message = ''
        if dataset_kind:
            dataset = self.get_data_by_kind(dataset_kind)
            if n_samples < dataset.number_of_images_cached:
                message = f'Sampling procedure took place, check used {n_samples} images from the original dataset.'
        else:
            if self._train is not None and n_samples < self._train.number_of_images_cached:
                message += f'Sampling procedure took place in <b>{self._train.name}</b> dataset, check used ' \
                           f'{n_samples} images from the original dataset.</br>'
            if self._test is not None and n_samples < self._test.number_of_images_cached:
                message += f'Sampling procedure took place in <b>{self._test.name}</b> dataset, check used ' \
                           f'{n_samples} images from the original dataset.</br>'

        if message:
            message = ('<p style="font-size:0.9em;line-height:1;"><i>'
                       f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" '
                       'parameter.</i></p>')
            check_result.display.append(message)
