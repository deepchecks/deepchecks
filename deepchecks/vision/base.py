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
# pylint: disable=broad-except,not-callable
from typing import Tuple, Mapping, Optional, Any, Union, Dict
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from ignite.metrics import Metric

from deepchecks.vision.utils.base_formatters import BasePredictionFormatter
from deepchecks.core.check import (
    CheckFailure,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck, CheckResult, BaseCheck, DatasetKind
)
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import (
    DatasetValidationError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)
from deepchecks.vision.dataset import VisionData, TaskType
from deepchecks.vision.utils.validation import apply_to_tensor
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter


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
    prediction_formatter : BasePredictionFormatter, default: None
        An encoder to convert predictions to a format that can be used by the metrics.
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
    """

    def __init__(self,
                 train: VisionData = None,
                 test: VisionData = None,
                 model: nn.Module = None,
                 model_name: str = '',
                 prediction_formatter: BasePredictionFormatter = None,
                 scorers: Mapping[str, Metric] = None,
                 scorers_per_class: Mapping[str, Metric] = None,
                 device: Union[str, torch.device, None] = 'cpu',
                 random_state: int = 42
                 ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if train and test:
            train.validate_shared_label(test)

        # Set seeds to if possible
        if train and random_state:
            train.set_seed(random_state)
        if test and random_state:
            test.set_seed(random_state)

        task_type = train.task_type if train else None
        self._device = torch.device(device) if isinstance(device, str) else (device if device else torch.device('cpu'))

        # If no prediction_formatter is passed and model and train are defined, we will use the default one according
        # to the dataset task type
        if model is not None and train is not None and prediction_formatter is None:
            if task_type == TaskType.CLASSIFICATION:
                prediction_formatter = ClassificationPredictionFormatter()
            elif task_type == TaskType.OBJECT_DETECTION:
                prediction_formatter = DetectionPredictionFormatter()
            else:
                raise DeepchecksValueError(f'Must pass prediction formatter for task_type {task_type}')

        if prediction_formatter is not None:
            if train is None or model is None:
                raise DeepchecksValueError('Can\'t pass prediction formatter without model and train data')
            prediction_formatter.validate_prediction(next(iter(train)), model, self._device)

        self._train = train
        self._test = test
        self._model = model
        self._batch_prediction_cache = None
        self._user_scorers = scorers
        self._user_scorers_per_class = scorers_per_class
        self._model_name = model_name
        self._prediction_formatter = prediction_formatter
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
    def prediction_formatter(self):
        """Return prediction formatter."""
        return self._prediction_formatter

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

    def infer(self, batch: Any) -> Any:
        """Return the predictions on the given batch, and cache them for later."""
        if self._batch_prediction_cache is None:
            self._batch_prediction_cache = self.prediction_formatter(batch, self.model, self.device)
        return self._batch_prediction_cache

    def flush_cached_inference(self):
        """Flush the cached inference."""
        self._batch_prediction_cache = None

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant VisionData by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')


def finalize_check_result(check_result: CheckResult, class_instance: BaseCheck) -> CheckResult:
    """Finalize the check result by adding the check instance and processing the conditions."""
    if not isinstance(check_result, CheckResult):
        raise DeepchecksValueError(f'Check {class_instance.name()} expected to return CheckResult but got: '
                                   + type(check_result).__name__)
    check_result.check = class_instance
    check_result.process_conditions()
    return check_result


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    def run(
        self,
        dataset: VisionData,
        model: Optional[nn.Module] = None,
        prediction_formatter: BasePredictionFormatter = None,
        device: Union[str, torch.device, None] = 'cpu',
        random_state: int = 42
    ) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(dataset,
                                    model=model,
                                    prediction_formatter=prediction_formatter,
                                    device=device,
                                    random_state=random_state)

        self.initialize_run(context, DatasetKind.TRAIN)

        for batch in dataset.get_data_loader():
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TRAIN)
            context.flush_cached_inference()

        return finalize_check_result(self.compute(context, DatasetKind.TRAIN), self)

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Update internal check state with given batch."""
        raise NotImplementedError()

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute final check result based on accumulated internal state."""
        raise NotImplementedError()


class TrainTestCheck(TrainTestBaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type = Context

    def run(
        self,
        train_dataset: VisionData,
        test_dataset: VisionData,
        model: Optional[nn.Module] = None,
        prediction_formatter: BasePredictionFormatter = None,
        device: Union[str, torch.device, None] = 'cpu',
        random_state: int = 42
    ) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(train_dataset,
                                    test_dataset,
                                    model=model,
                                    prediction_formatter=prediction_formatter,
                                    device=device,
                                    random_state=random_state)

        self.initialize_run(context)

        for batch in context.train.get_data_loader():
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TRAIN)
            context.flush_cached_inference()

        for batch in context.test.get_data_loader():
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TEST)
            context.flush_cached_inference()

        return finalize_check_result(self.compute(context), self)

    def initialize_run(self, context: Context):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Update internal check state with given batch for either train or test."""
        raise NotImplementedError()

    def compute(self, context: Context) -> CheckResult:
        """Compute final check result based on accumulated internal state."""
        raise NotImplementedError()


class ModelOnlyCheck(ModelOnlyBaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type = Context

    def run(
        self,
        model: nn.Module,
        device: Union[str, torch.device, None] = 'cpu',
        random_state: int = 42
    ) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context = self.context_type(model=model, device=device, random_state=random_state)

        self.initialize_run(context)
        return finalize_check_result(self.compute(context), self)

    def initialize_run(self, context: Context):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def compute(self, context: Context) -> CheckResult:
        """Compute final check result."""
        raise NotImplementedError()


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
            prediction_formatter: BasePredictionFormatter = None,
            scorers: Mapping[str, Metric] = None,
            scorers_per_class: Mapping[str, Metric] = None,
            device: Union[str, torch.device, None] = 'cpu',
            random_state: int = 42
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
        prediction_formatter : BasePredictionFormatter, default: None
            An encoder to convert predictions to a format that can be used by the metrics.
        scorers : Mapping[str, Metric] , default None
            dict of scorers names to scorer sklearn_name/function
        scorers_per_class : Mapping[str, Metric], default None
            dict of scorers for classification without averaging of the classes
            See <a href=
            "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
            scikit-learn docs</a>
        device : Union[str, torch.device], default: None
            processing unit for use
        random_state : int
            A seed to set for pseudo-random functions

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = Context(
            train_dataset,
            test_dataset,
            model,
            prediction_formatter=prediction_formatter,
            scorers=scorers,
            scorers_per_class=scorers_per_class,
            device=device,
            random_state=random_state
        )

        # Create instances of SingleDatasetCheck for train and test if train and test exist.
        # This is needed because in the vision package checks update their internal state with update, so it will be
        # easier to iterate and keep the check order if we have an instance for each dataset.
        checks: Dict[
            Union[str, int],
            Union[SingleDatasetCheck, TrainTestCheck, ModelOnlyCheck]
        ] = OrderedDict({})

        for check_idx, check in list(self.checks.items()):
            if isinstance(check, (TrainTestCheck, ModelOnlyCheck)):
                check.initialize_run(context)
                checks[check_idx] = check
            elif isinstance(check, SingleDatasetCheck):
                if train_dataset is not None:
                    checks[str(check_idx) + ' - Train'] = check
                if test_dataset is not None:
                    checks[str(check_idx) + ' - Test'] = check

        results: Dict[
            Union[str, int],
            Union[CheckResult, CheckFailure]
        ] = OrderedDict({})

        run_train_test_checks = train_dataset is not None and test_dataset is not None

        if train_dataset is not None:
            self._update_loop(
                checks=checks,
                data_loader=train_dataset.get_data_loader(),
                context=context,
                run_train_test_checks=run_train_test_checks,
                results=results,
                dataset_kind=DatasetKind.TRAIN
            )
            for check_idx, check in checks.items():
                if check_idx not in results:
                    if str(check_idx).endswith('Train'):
                        try:
                            results[check_idx] = check.compute(context, dataset_kind=DatasetKind.TRAIN)
                        except Exception as exp:
                            results[check_idx] = CheckFailure(check, exp, ' - Train')

        if test_dataset is not None:
            self._update_loop(
                checks=checks,
                data_loader=test_dataset.get_data_loader(),
                context=context,
                run_train_test_checks=run_train_test_checks,
                results=results,
                dataset_kind=DatasetKind.TEST
            )
            for check_idx, check in checks.items():
                if check_idx not in results:
                    if str(check_idx).endswith('Test'):
                        try:
                            results[check_idx] = check.compute(context, dataset_kind=DatasetKind.TEST)
                        except Exception as exp:
                            results[check_idx] = CheckFailure(check, exp, ' - Test')

        for check_idx, check in checks.items():
            if check_idx not in results:
                try:
                    if not isinstance(check, SingleDatasetCheck):
                        results[check_idx] = check.compute(context)
                except Exception as exp:
                    results[check_idx] = CheckFailure(check, exp)

        # Update check result names for SingleDatasetChecks and finalize results
        for check_idx, result in results.items():
            if isinstance(result, CheckResult):
                result = finalize_check_result(result, checks[check_idx])
                result.header = (
                    f'{result.get_header()} - Train Dataset'
                    if str(check_idx).endswith(' - Train')
                    else f'{result.get_header()} - Test Dataset'
                )
                results[check_idx] = result

        return SuiteResult(self.name, list(results.values()))

    def _update_loop(
        self,
        checks: Dict[
            Union[str, int],
            Union[SingleDatasetCheck, TrainTestCheck, ModelOnlyCheck]
        ],
        data_loader: DataLoader,
        context: Context,
        run_train_test_checks: bool,
        results: Dict[Union[str, int], Union[CheckResult, CheckFailure]],
        dataset_kind
    ):
        if dataset_kind == DatasetKind.TEST:
            type_suffix = ' - Test'
        else:
            type_suffix = ' - Train'
        n_batches = len(data_loader)
        progress_bar = ProgressBar(self.name + type_suffix, n_batches)

        for idx, check in checks.items():
            if str(idx).endswith(type_suffix):
                check.initialize_run(context, dataset_kind=dataset_kind)

        for batch_id, batch in enumerate(data_loader):
            progress_bar.set_text(f'{100 * batch_id / (1. * n_batches):.0f}%')
            batch = apply_to_tensor(batch, lambda it: it.to(context.device))
            for check_idx, check in checks.items():
                try:
                    if isinstance(check, TrainTestCheck):
                        if run_train_test_checks is True:
                            check.update(context, batch, dataset_kind=dataset_kind)
                        else:
                            msg = 'Check is irrelevant if not supplied with both train and test datasets'
                            results[check_idx] = self._get_unsupported_failure(check, msg)
                    elif isinstance(check, SingleDatasetCheck):
                        if str(check_idx).endswith(type_suffix):
                            check.update(context, batch, dataset_kind=dataset_kind)
                    elif isinstance(check, ModelOnlyCheck):
                        pass
                    else:
                        raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
                except Exception as exp:
                    results[check_idx] = CheckFailure(check, exp, type_suffix)
            progress_bar.inc_progress()
            context.flush_cached_inference()

        progress_bar.close()

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
