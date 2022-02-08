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
"""Module for base vision abstractions."""
# TODO: This file should be completely modified
# pylint: disable=broad-except,not-callable
from typing import Tuple, Mapping, Optional, Union

import torch
from torch import nn
from ignite.metrics import Metric

from deepchecks.vision.utils.validation import validate_model
from deepchecks.core.check import (
    CheckFailure,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
)
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import (
    DatasetValidationError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)
from deepchecks.vision.dataset import VisionData, TaskType


__all__ = [
    'Context',
    'Suite',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
]


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
    """

    def __init__(self,
                 train: VisionData = None,
                 test: VisionData = None,
                 model: nn.Module = None,
                 model_name: str = '',
                 scorers: Mapping[str, Metric] = None,
                 scorers_per_class: Mapping[str, Metric] = None,
                 device: Union[str, torch.device, None] = None
                 ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if train and test:
            train.validate_shared_label(test)

        self._train = train
        self._test = test
        self._model = model
        self._validated_model = False
        self._train_sample_predictions = None
        self._test_sample_predictions = None
        self._user_scorers = scorers
        self._user_scorers_per_class = scorers_per_class
        self._model_name = model_name
        self._device = torch.device(device) if isinstance(device, str) else device

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
        if not self._validated_model:
            if self._train:
                validate_model(self._train, self._model)
            self._validated_model = True
        return self._model

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name

    @property
    def train_sample_predictions(self):
        """Return the predictions on the train samples."""
        if self._train_sample_predictions is None:
            self._train_sample_predictions = []
            for tensor, _ in self.train.sample_data_loader:
                self._train_sample_predictions.append(self.model(tensor))
        return self._train_sample_predictions

    @property
    def test_sample_predictions(self):
        """Return the predictions on the test samples."""
        if self._test_sample_predictions is None:
            self._test_sample_predictions = []
            for tensor, _ in self.test.sample_data_loader:
                self._test_sample_predictions.append(self.model(tensor))
        return self._test_sample_predictions

    @property
    def device(self) -> Optional[torch.device]:
        """Return device to use specified by the user."""
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


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context


class TrainTestCheck(TrainTestBaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type = Context


class ModelOnlyCheck(ModelOnlyBaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type = Context


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck

    def run(
            self,
            train_dataset: Optional[VisionData] = None,
            test_dataset: Optional[VisionData] = None,
            model: nn.Module = None,
            scorers: Mapping[str, Metric] = None,
            scorers_per_class: Mapping[str, Metric] = None
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Optional[VisionData] , default None
            object, representing data an estimator was fitted on
        test_dataset : Optional[VisionData] , default None
            object, representing data an estimator predicts on
        model : nn.Module , default None
            A scikit-learn-compatible fitted estimator instance
        scorers : Mapping[str, Metric] , default None
            dict of scorers names to scorer sklearn_name/function
        scorers_per_class : Mapping[str, Metric], default None
            dict of scorers for classification without averaging of the classes
            See <a href=
            "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
            scikit-learn docs</a>
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = Context(train_dataset, test_dataset, model,
                          scorers=scorers,
                          scorers_per_class=scorers_per_class)
        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if not supplied with both train and test datasets'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, SingleDatasetCheck):
                    if train_dataset is not None:
                        # In case of train & test, doesn't want to skip test if train fails. so have to explicitly
                        # wrap it in try/except
                        try:
                            check_result = check.run_logic(context)
                            # In case of single dataset not need to edit the header
                            if test_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Train Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Train Dataset')
                        results.append(check_result)
                    if test_dataset is not None:
                        try:
                            check_result = check.run_logic(context, dataset_type='test')
                            # In case of single dataset not need to edit the header
                            if train_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Test Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Test Dataset')
                        results.append(check_result)
                    if train_dataset is None and test_dataset is None:
                        msg = 'Check is irrelevant if dataset is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, ModelOnlyCheck):
                    if model is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if model is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
