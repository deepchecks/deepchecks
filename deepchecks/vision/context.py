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
from typing import Mapping, Union, Iterable, Any, Tuple

import torch
from torch import nn
from ignite.metrics import Metric

from deepchecks.core import DatasetKind
from deepchecks.vision.vision_data import VisionData, TaskType
from deepchecks.vision.utils.validation import apply_to_tensor
from deepchecks.core.errors import (
    DatasetValidationError, DeepchecksNotImplementedError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)


__all__ = ['Context', 'Batch']


logger = logging.getLogger('deepchecks')


class Batch:
    """Represents dataset batch returned by the dataloader during iteration."""

    def __init__(
        self,
        batch: Tuple[Iterable[Any], Iterable[Any]],
        context: 'Context',
        dataset_kind: DatasetKind
    ):
        self._context = context
        self._dataset_kind = dataset_kind
        self._batch = apply_to_tensor(batch, lambda it: it.to(self._context.device))
        self._labels = None
        self._predictions = None
        self._images = None

    @property
    def labels(self):
        """Return labels for the batch, formatted in deepchecks format."""
        if self._labels is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            self._labels = dataset.batch_to_labels(self._batch)
        return self._labels

    @property
    def predictions(self):
        """Return predictions for the batch, formatted in deepchecks format."""
        if self._predictions is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            self._predictions = dataset.infer_on_batch(self._batch, self._context.model, self._context.device)
        return self._predictions

    @property
    def images(self):
        """Return images for the batch, formatted in deepchecks format."""
        if self._images is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            self._images = dataset.batch_to_images(self._batch)
        return self._images

    def __getitem__(self, index):
        """Return batch item by index."""
        return self._batch[index]


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

        if model is not None:
            if not isinstance(model, nn.Module):
                logger.warning('Model is not a torch.nn.Module. Deepchecks can\'t validate that model is in '
                               'evaluation state.')
            else:
                if model.training:
                    raise DatasetValidationError('Model is not in evaluation state. Please set model training '
                                                 'parameter to False or run model.eval() before passing it.')
            for dataset, dataset_type in zip([train, test], ['train', 'test']):
                if dataset is not None:
                    try:
                        dataset.validate_prediction(next(iter(dataset.data_loader)), model, self._device)
                    except DeepchecksNotImplementedError:
                        logger.warning('validate_prediction() was not implemented in %s dataset, '
                                       'some checks will not run', dataset_type)

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

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant VisionData by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')
