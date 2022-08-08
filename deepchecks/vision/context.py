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
from operator import itemgetter
from typing import Dict, Mapping, Optional, Sequence, Union

import torch
from ignite.metrics import Metric
from torch import nn

from deepchecks.core import CheckFailure, CheckResult, DatasetKind, SuiteResult
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotImplementedError, DeepchecksNotSupportedError,
                                    DeepchecksValueError, ModelValidationError, ValidationError)
from deepchecks.utils.logger import get_logger
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.vision_properties import STATIC_PROPERTIES_FORMAT
from deepchecks.vision.vision_data import VisionData

__all__ = ['Context']


@docstrings
class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train : Optional[VisionData] , default: None
        VisionData object, representing data an neural network was fitted on
    test : Optional[VisionData] , default: None
        VisionData object, representing data an neural network predicts on
    model : Optional[nn.Module] , default: None
        pytorch neural network module instance
    {additional_context_params:indent}
    """

    def __init__(
        self,
        train: Optional[VisionData] = None,
        test: Optional[VisionData] = None,
        model: Optional[nn.Module] = None,
        model_name: str = '',
        scorers: Optional[Mapping[str, Metric]] = None,
        scorers_per_class: Optional[Mapping[str, Metric]] = None,
        device: Union[str, torch.device, None] = None,
        random_state: int = 42,
        n_samples: Optional[int] = None,
        with_display: bool = True,
        train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        train_properties: Optional[STATIC_PROPERTIES_FORMAT] = None,
        test_properties: Optional[STATIC_PROPERTIES_FORMAT] = None
    ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if train and test:
            train.validate_shared_label(test)

        if device is None:
            device = 'cpu'
            if torch.cuda.is_available():
                get_logger().warning('Checks will run on the cpu by default. To make use of cuda devices, '
                                     'use the device parameter in the run function.')
        self._device = torch.device(device) if isinstance(device, str) else (device if device else torch.device('cpu'))

        self._prediction_formatter_error = {}
        if model is not None:
            self._static_predictions = None
            if not isinstance(model, nn.Module):
                get_logger().warning('Deepchecks can\'t validate that model is in evaluation state.'
                                     ' Make sure it is to avoid unexpected behavior.')
            elif model.training:
                raise DatasetValidationError('Model is not in evaluation state. Please set model training '
                                             'parameter to False or run model.eval() before passing it.')

            for dataset, dataset_type in zip([train, test], [DatasetKind.TRAIN, DatasetKind.TEST]):
                if dataset is not None:
                    try:
                        dataset.validate_prediction(next(iter(dataset.data_loader)), model, self._device)
                        msg = None
                    except DeepchecksNotImplementedError:
                        msg = f'infer_on_batch() was not implemented in {dataset_type} ' \
                           f'dataset, some checks will not run'
                    except ValidationError as ex:
                        msg = f'infer_on_batch() was not implemented correctly in the {dataset_type} dataset, the ' \
                           f'validation has failed with the error: {ex}. To test your prediction formatting use the ' \
                           'function `vision_data.validate_prediction(batch, model, device)`'

                    if msg:
                        self._prediction_formatter_error[dataset_type] = msg
                        get_logger().warning(msg)

        elif train_predictions is not None or test_predictions is not None:
            self._static_predictions = {}
            for dataset, dataset_type, predictions in zip([train, test],
                                                          [DatasetKind.TRAIN, DatasetKind.TEST],
                                                          [train_predictions, test_predictions]):
                if dataset is not None:
                    try:
                        preds = itemgetter(*list(dataset.data_loader.batch_sampler)[0])(predictions)
                        if dataset.task_type == TaskType.CLASSIFICATION:
                            preds = torch.stack(preds)
                        dataset.validate_infered_batch_predictions(preds)
                        msg = None
                        self._static_predictions[dataset_type] = predictions
                    except ValidationError as ex:
                        msg = f'the predictions given were not in a correct format in the {dataset_type} dataset, ' \
                            f'the validation has failed with the error: {ex}. To test your prediction formatting' \
                            ' use the function `vision_data.validate_inferred_batch_predictions(predictions)`'

                    if msg:
                        self._prediction_formatter_error[dataset_type] = msg
                        get_logger().warning(msg)
        self._static_properties = None
        if train_properties is not None or test_properties is not None:
            self._static_properties = {}
            for dataset, dataset_type, properties in zip([train, test],
                                                         [DatasetKind.TRAIN, DatasetKind.TEST],
                                                         [train_properties, test_properties]):
                if dataset is not None:
                    try:
                        props = itemgetter(*list(dataset.data_loader.batch_sampler)[0])(properties)
                        msg = None
                        self._static_properties[dataset_type] = props
                    except ValidationError as ex:
                        msg = f'the properties given were not in a correct format in the {dataset_type} dataset, ' \
                            f'the validation has failed with the error: {ex}.'

        # The copy does 2 things: Sample n_samples if parameter exists, and shuffle the data.
        # we shuffle because the data in VisionData is set to be sampled in a fixed order (in the init), so if the user
        # wants to run without random_state we need to forcefully shuffle (to have different results on different runs
        # from the same VisionData object), and if there is a random_state the shuffle will always have same result
        if train:
            train = train.copy(shuffle=True, n_samples=n_samples, random_state=random_state)
        if test:
            test = test.copy(shuffle=True, n_samples=n_samples, random_state=random_state)

        self._train = train
        self._test = test
        self._model = model
        self._user_scorers = scorers
        self._user_scorers_per_class = scorers_per_class
        self._model_name = model_name
        self._with_display = with_display
        self.random_state = random_state

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

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

    @property
    def model(self) -> nn.Module:
        """Return & validate model if model exists, otherwise raise error."""
        if self._model is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without model')
        return self._model

    @property
    def static_predictions(self) -> Dict:
        """Return the static_predictions."""
        return self._static_predictions

    @property
    def static_properties(self) -> Dict:
        """Return the static_predictions."""
        return self._static_properties

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name

    @property
    def device(self) -> torch.device:
        """Return device specified by the user."""
        return self._device

    def have_test(self):
        """Return whether there is test dataset defined."""
        return self._test is not None

    def assert_task_type(self, *expected_types: TaskType):
        """Assert task_type matching given types."""
        if self.train.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is irrelevant for task of type {self.train.task_type}')
        return True

    def assert_predictions_valid(self, kind: DatasetKind = None):
        """Assert that for given DatasetKind the model & dataset infer_on_batch return predictions in right format."""
        error = self._prediction_formatter_error.get(kind)
        if error:
            raise DeepchecksValueError(error)

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant VisionData by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')

    def add_is_sampled_footnote(self, result: Union[CheckResult, SuiteResult], kind: DatasetKind = None):
        """Get footnote to display when the datasets are sampled."""
        message = ''
        if kind:
            v_data = self.get_data_by_kind(kind)
            if v_data.is_sampled():
                message = f'Data is sampled from the original dataset, running on {v_data.num_samples} samples out of' \
                          f' {v_data.original_num_samples}.'
        else:
            if self._train is not None and self._train.is_sampled():
                message += f'Running on {self._train.num_samples} <b>train</b> data samples out of ' \
                           f'{self._train.original_num_samples}.'
            if self._test is not None and self._test.is_sampled():
                if message:
                    message += ' '
                message += f'Running on {self._test.num_samples} <b>test</b> data samples out of ' \
                           f'{self._test.original_num_samples}.'

        if message:
            message = ('<p style="font-size:0.9em;line-height:1;"><i>'
                       f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" parameter.'
                       '</i></p>')
            if isinstance(result, CheckResult):
                result.display.append(message)
            elif isinstance(result, SuiteResult):
                result.extra_info.append(message)

    def finalize_check_result(self, check_result, check):
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
