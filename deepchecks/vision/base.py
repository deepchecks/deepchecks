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
    DeepchecksNotSupportedError, DeepchecksValueError
)
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.utils.validation import apply_to_tensor

logger = logging.getLogger('deepchecks')

__all__ = [
    'Suite',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
]

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

        results: Dict[
            Union[str, int],
            Union[CheckResult, CheckFailure]
        ] = OrderedDict({})

        run_train_test_checks = train_dataset is not None and test_dataset is not None

        # Initialize here all the checks that are not single dataset, since those are initialized inside the update loop
        for index, check in self.checks.items():
            if not isinstance(check, SingleDatasetCheck):
                try:
                    check.initialize_run(context)
                except Exception as exp:
                    results[index] = CheckFailure(check, exp)

        if train_dataset is not None:
            self._update_loop(
                context=context,
                run_train_test_checks=run_train_test_checks,
                results=results,
                dataset_kind=DatasetKind.TRAIN
            )

        if test_dataset is not None:
            self._update_loop(
                context=context,
                run_train_test_checks=run_train_test_checks,
                results=results,
                dataset_kind=DatasetKind.TEST
            )

        for check_idx, check in self.checks.items():
            try:
                # if check index in results we had failure, and SingleDatasetCheck have already been calculated inside
                # the loops
                if check_idx not in results and not isinstance(check, SingleDatasetCheck):
                    result = check.compute(context)
                    result = finalize_check_result(result, check)
                    results[check_idx] = result
            except Exception as exp:
                results[check_idx] = CheckFailure(check, exp)

        # The results are ordered as they ran instead of in the order they were defined, therefore sort by key
        sorted_result_values = [value for name, value in sorted(results.items(), key=lambda pair: str(pair[0]))]
        return SuiteResult(self.name, sorted_result_values)

    def _update_loop(
        self,
        context: Context,
        run_train_test_checks: bool,
        results: Dict[Union[str, int], Union[CheckResult, CheckFailure]],
        dataset_kind: DatasetKind
    ):
        type_suffix = ' - Test Dataset' if dataset_kind == DatasetKind.TEST else ' - Train Dataset'
        data_loader = context.get_data_by_kind(dataset_kind)
        n_batches = len(data_loader)
        progress_bar = ProgressBar(self.name + type_suffix, n_batches)

        # SingleDatasetChecks have different handling, need to initialize them here (to have them ready for different
        # dataset kind)
        for idx, check in self.checks.items():
            if isinstance(check, SingleDatasetCheck):
                try:
                    check.initialize_run(context, dataset_kind=dataset_kind)
                except Exception as exp:
                    results[idx] = CheckFailure(check, exp, type_suffix)

        for batch_id, batch in enumerate(data_loader):
            progress_bar.set_text(f'{100 * batch_id / (1. * n_batches):.0f}%')
            batch = apply_to_tensor(batch, lambda it: it.to(context.device))
            for check_idx, check in self.checks.items():
                # If index in results the check already failed before
                if check_idx in results:
                    continue
                try:
                    if isinstance(check, TrainTestCheck):
                        if run_train_test_checks is True:
                            check.update(context, batch, dataset_kind=dataset_kind)
                        else:
                            msg = 'Check is irrelevant if not supplied with both train and test datasets'
                            results[check_idx] = self._get_unsupported_failure(check, msg)
                    elif isinstance(check, SingleDatasetCheck):
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

        # SingleDatasetChecks have different handling. If we had failure in them need to add suffix to the index of
        # the results, else need to compute it.
        for idx, check in self.checks.items():
            if isinstance(check, SingleDatasetCheck):
                index_of_kind = str(idx) + type_suffix
                # If index in results we had a failure
                if idx in results:
                    results[index_of_kind] = results.pop(idx)
                    continue
                try:
                    result = check.compute(context, dataset_kind=dataset_kind)
                    result = finalize_check_result(result, check)
                    # Update header with dataset type only if both train and test ran
                    if run_train_test_checks:
                        result.header = result.get_header() + type_suffix
                    results[index_of_kind] = result
                except Exception as exp:
                    results[index_of_kind] = CheckFailure(check, exp, type_suffix)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
