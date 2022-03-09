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
import logging
from typing import Tuple, Mapping, Optional, Any, Union, Dict
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from ignite.metrics import Metric

from deepchecks.core.check import (
    CheckFailure,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck, CheckResult, BaseCheck, DatasetKind
)
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import (
    DatasetValidationError, DeepchecksNotImplementedError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)
from deepchecks.vision.vision_data import VisionData, TaskType
from deepchecks.vision.utils.validation import apply_to_tensor

logger = logging.getLogger('deepchecks')

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

        self._device = torch.device(device) if isinstance(device, str) else (device if device else torch.device('cpu'))

        if model is not None:
            for dataset, dataset_type in zip([train, test], ['train', 'test']):
                if dataset is not None:
                    try:
                        dataset.validate_prediction(next(iter(dataset.data_loader)), model, self._device)
                    except DeepchecksNotImplementedError:
                        logger.warning('validate_prediction() was not implemented in %s dataset, '
                                       'some checks will not run', dataset_type)

        # Set seeds to if possible (we set it after the validation to keep the seed state)
        if train and random_state:
            train.set_seed(random_state)
        if test and random_state:
            test.set_seed(random_state)

        self._train = train
        self._test = test
        self._model = model
        self._batch_prediction_cache = {}
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

    def infer(self, batch: Any, dataset_kind: DatasetKind) -> Any:
        """Return the predictions on the given batch, and cache them for later."""
        if self._batch_prediction_cache.get(dataset_kind) is None:
            dataset = self.get_data_by_kind(dataset_kind)
            self._batch_prediction_cache[dataset_kind] = dataset.infer_on_batch(batch, self.model, self.device)
        return self._batch_prediction_cache[dataset_kind]

    def flush_cached_inference(self, dataset_kind: DatasetKind):
        """Flush the cached inference."""
        self._batch_prediction_cache[dataset_kind] = None

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
        device: Union[str, torch.device, None] = 'cpu',
        random_state: int = 42
    ) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context: Context = self.context_type(dataset,
                                             model=model,
                                             device=device,
                                             random_state=random_state)

        self.initialize_run(context, DatasetKind.TRAIN)

        for batch in dataset:
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TRAIN)
            context.flush_cached_inference(DatasetKind.TRAIN)

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
        device: Union[str, torch.device, None] = 'cpu',
        random_state: int = 42
    ) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        context: Context = self.context_type(train_dataset,
                                             test_dataset,
                                             model=model,
                                             device=device,
                                             random_state=random_state)

        self.initialize_run(context)

        for batch in context.train:
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TRAIN)
            context.flush_cached_inference(DatasetKind.TRAIN)

        for batch in context.test:
            batch = apply_to_tensor(batch, lambda x: x.to(device))
            self.update(context, batch, DatasetKind.TEST)
            context.flush_cached_inference(DatasetKind.TEST)

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
        context: Context = self.context_type(model=model, device=device, random_state=random_state)

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

        results: Dict[
            Union[str, int],
            Union[CheckResult, CheckFailure]
        ] = OrderedDict({})

        for check_idx, check in list(self.checks.items()):
            if isinstance(check, (TrainTestCheck, ModelOnlyCheck)):
                try:
                    check.initialize_run(context)
                except Exception as exp:
                    results[check_idx] = CheckFailure(check, exp)
                checks[check_idx] = check
            elif isinstance(check, SingleDatasetCheck):
                if train_dataset is not None:
                    checks[str(check_idx) + ' - Train'] = check
                if test_dataset is not None:
                    checks[str(check_idx) + ' - Test'] = check
            else:
                raise DeepchecksNotSupportedError(f'Don\'t know to handle check type {type(check)}')

        run_train_test_checks = train_dataset is not None and test_dataset is not None

        if train_dataset is not None:
            self._update_loop(
                checks=checks,
                data_loader=train_dataset.data_loader,
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
                data_loader=test_dataset.data_loader,
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
                results[check_idx] = result
                # Update header only if both train and test ran
                if run_train_test_checks:
                    result.header = (
                        f'{result.get_header()} - Train Dataset'
                        if str(check_idx).endswith(' - Train')
                        else f'{result.get_header()} - Test Dataset'
                    )

        # The results are ordered as they ran instead of in the order they were defined, therefore sort by key
        sorted_result_values = [value for name, value in sorted(results.items(), key=lambda pair: str(pair[0]))]
        return SuiteResult(self.name, sorted_result_values)

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
                try:
                    check.initialize_run(context, dataset_kind=dataset_kind)
                except Exception as exp:
                    results[idx] = CheckFailure(check, exp, type_suffix)

        for batch_id, batch in enumerate(data_loader):
            progress_bar.set_text(f'{100 * batch_id / (1. * n_batches):.0f}%')
            batch = apply_to_tensor(batch, lambda it: it.to(context.device))
            for check_idx, check in checks.items():
                if results.get(check_idx):
                    continue
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
            context.flush_cached_inference(dataset_kind)

        progress_bar.close()

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
