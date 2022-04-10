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
import logging
from typing import Mapping, Union

import torch
from torch import nn
from ignite.metrics import Metric

from deepchecks.core import DatasetKind
from deepchecks.vision.vision_data import VisionData, TaskType
from deepchecks.core.errors import (
    DatasetValidationError, DeepchecksNotImplementedError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError, ValidationError
)


__all__ = ['Context']


logger = logging.getLogger('deepchecks')


class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train : VisionData , default: None
        Dataset or DataFrame object, representing data an estimator was fitted on
    test : VisionData , default: None
        Dataset or DataFrame object, representing data an estimator predicts on
    model : BasicModel , default: None
        A scikit-learn-compatible fitted estimator instance
    model_name: str , default: ''
        The name of the model
    scorers : Mapping[str, Metric] , default: None
        dict of scorers names to a Metric
    scorers_per_class : Mapping[str, Metric] , default: None
        dict of scorers for classification without averaging of the classes.
        See <a href=
        "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
        scikit-learn docs</a>
    device : Union[str, torch.device], default: 'cpu'
        processing unit for use
    random_state : int
        A seed to set for pseudo-random functions
    n_samples : int, default: None
    """

    def __init__(self,
                 train: VisionData = None,
                 test: VisionData = None,
                 model: nn.Module = None,
                 model_name: str = '',
                 scorers: Mapping[str, Metric] = None,
                 scorers_per_class: Mapping[str, Metric] = None,
                 device: Union[str, torch.device, None] = 'cpu',
                 random_state: int = 42,
                 n_samples: int = None
                 ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if train and test:
            train.validate_shared_label(test)

        self._device = torch.device(device) if isinstance(device, str) else (device if device else torch.device('cpu'))
        self._prediction_formatter_error = {}

        if model is not None:
            if not isinstance(model, nn.Module):
                logger.warning('Model is not a torch.nn.Module. Deepchecks can\'t validate that model is in '
                               'evaluation state.')
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
                        msg = f'infer_on_batch() was not implemented correctly in {dataset_type}, the ' \
                           f'validation has failed with the error: {ex}. To test your prediction formatting use the ' \
                           f'function `vision_data.validate_prediction(batch, model, device)`'

                    if msg:
                        self._prediction_formatter_error[dataset_type] = msg
                        logger.warning(msg)

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
        self.random_state = random_state

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

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

    def get_is_sampled_footnote(self, kind: DatasetKind = None):
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
            message = f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" parameter.'
            return f'<p style="font-size:0.9em;line-height:1;"><i>{message}</i></p>'
