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
"""Module for base context superclasses."""


from abc import ABC, abstractmethod

from deepchecks.core import CheckFailure, CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError


class BaseContext(ABC):
    """Base class for contexts."""

    _train = None
    _test = None
    _model = None
    _with_display: bool = True

    @property
    def with_display(self) -> bool:
        """Return the with_display flag."""
        return self._with_display

    @property
    def train(self):
        """Return train if exists, otherwise raise error."""
        if self._train is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self):
        """Return test if exists, otherwise raise error."""
        if self._test is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without test dataset')
        return self._test

    def assert_task_type(self, *expected_types):
        """Assert task_type matching given types.

        If task_type is defined, validate it and raise error if needed, else returns True.
        If task_type is not defined, return False.
        """
        if self.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{self.task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )
        return True

    @property
    @abstractmethod
    def task_type(self):
        """Return the task type."""
        raise NotImplementedError()

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant Dataset by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')

    def finalize_check_result(self, check_result, check, dataset_kind: DatasetKind = None):
        """Run final processing on a check result which includes validation, conditions processing and sampling\
        footnote."""
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
        if hasattr(check, 'n_samples'):
            n_samples = getattr(check, 'n_samples')
            message = ''
            if dataset_kind:
                dataset = self.get_data_by_kind(dataset_kind)
                if dataset.is_sampled(n_samples):
                    message = f'Data is sampled from the original dataset, running on ' \
                              f'{dataset.len_when_sampled(n_samples)} samples out of {len(dataset)}.'
            else:
                if self._train is not None and self._train.is_sampled(n_samples):
                    message += f'Running on {self._train.len_when_sampled(n_samples)} <b>train</b> data samples ' \
                               f'out of {len(self._train)}.'
                if self._test is not None and self._test.is_sampled(n_samples):
                    if message:
                        message += ' '
                    message += f'Running on {self._test.len_when_sampled(n_samples)} <b>test</b> data samples ' \
                               f'out of {len(self._test)}.'

            if message:
                message = ('<p style="font-size:0.9em;line-height:1;"><i>'
                           f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" '
                           'parameter.</i></p>')
                check_result.display.append(message)
