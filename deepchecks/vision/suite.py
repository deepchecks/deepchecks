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
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from ignite.metrics import Metric
from torch import nn

from deepchecks.core.check_result import BaseCheckResult, CheckFailure
from deepchecks.core.checks import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.utils.ipython import ProgressBarGroup
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData

__all__ = ['Suite']


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck

    @docstrings
    def run(
        self,
        train_dataset: Optional[VisionData] = None,
        test_dataset: Optional[VisionData] = None,
        model: Optional[nn.Module] = None,
        scorers: Optional[Mapping[str, Metric]] = None,
        scorers_per_class: Optional[Mapping[str, Metric]] = None,
        device: Union[str, torch.device, None] = None,
        random_state: int = 42,
        with_display: bool = True,
        n_samples: Optional[int] = None,
        train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        model_name: str = '',
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
        {additional_context_params:2*indent}

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        run_train_test_checks = train_dataset is not None and test_dataset is not None
        non_single_checks = {k: check for k, check in self.checks.items() if not isinstance(check, SingleDatasetCheck)}

        results: Dict[
            Union[str, int],
            BaseCheckResult
        ] = OrderedDict({})

        with ProgressBarGroup() as progressbar_factory:

            with progressbar_factory.create_dummy(name='Validating Input'):
                context = Context(
                    train_dataset,
                    test_dataset,
                    model,
                    scorers=scorers,
                    scorers_per_class=scorers_per_class,
                    device=device,
                    random_state=random_state,
                    n_samples=n_samples,
                    with_display=with_display,
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    model_name=model_name
                )

            # Initialize here all the checks that are not single dataset,
            # since those are initialized inside the update loop
            for index, check in non_single_checks.items():
                try:
                    check.initialize_run(context)
                except Exception as exp:
                    results[index] = CheckFailure(check, exp)

            if train_dataset is not None:
                self._update_loop(
                    context=context,
                    run_train_test_checks=run_train_test_checks,
                    results=results,
                    dataset_kind=DatasetKind.TRAIN,
                    progressbar_factory=progressbar_factory
                )

            if test_dataset is not None:
                self._update_loop(
                    context=context,
                    run_train_test_checks=run_train_test_checks,
                    results=results,
                    dataset_kind=DatasetKind.TEST,
                    progressbar_factory=progressbar_factory
                )

            # Need to compute only on not SingleDatasetCheck, since they computed inside the loop
            if non_single_checks:
                progress_bar = progressbar_factory.create(
                    iterable=list(non_single_checks.items()),
                    name='Computing Checks',
                    unit='Check'
                )
                for check_idx, check in progress_bar:
                    progress_bar.set_postfix({'Check': check.name()})
                    try:
                        # if check index in results we had failure
                        if check_idx not in results:
                            result = check.compute(context)
                            context.finalize_check_result(result, check)
                            results[check_idx] = result
                    except Exception as exp:
                        results[check_idx] = CheckFailure(check, exp)

        # The results are ordered as they ran instead of in the order they were defined, therefore sort by key
        sorted_result_values = [value for name, value in sorted(results.items(), key=lambda pair: str(pair[0]))]

        result = SuiteResult(self.name, sorted_result_values)
        context.add_is_sampled_footnote(result)
        return result

    def _update_loop(
        self,
        context: Context,
        run_train_test_checks: bool,
        results: Dict[Union[str, int], BaseCheckResult],
        dataset_kind: DatasetKind,
        progressbar_factory: ProgressBarGroup
    ):
        type_suffix = ' - Test Dataset' if dataset_kind == DatasetKind.TEST else ' - Train Dataset'
        vision_data = context.get_data_by_kind(dataset_kind)
        single_dataset_checks = {k: check for k, check in self.checks.items() if isinstance(check, SingleDatasetCheck)}

        # SingleDatasetChecks have different handling, need to initialize them here (to have them ready for different
        # dataset kind)
        for idx, check in single_dataset_checks.items():
            try:
                check.initialize_run(context, dataset_kind=dataset_kind)
            except Exception as exp:
                results[idx] = CheckFailure(check, exp, type_suffix)

        # Init cache of vision_data
        vision_data.init_cache()

        batches_pbar = progressbar_factory.create(
            iterable=vision_data,
            name='Ingesting Batches' + type_suffix,
            unit='Batch'
        )

        # Run on all the batches
        for i, batch in enumerate(batches_pbar):
            batch = Batch(batch, context, dataset_kind, i)
            vision_data.update_cache(batch)
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

        # SingleDatasetChecks have different handling. If we had failure in them need to add suffix to the index of
        # the results, else need to compute it.
        if single_dataset_checks:
            checks_pbar = progressbar_factory.create(
                iterable=list(single_dataset_checks.items()),
                name='Computing Single Dataset Checks' + type_suffix,
                unit='Check'
            )
            for idx, check in checks_pbar:
                checks_pbar.set_postfix({'Check': check.name()}, refresh=False)
                index_of_kind = str(idx) + type_suffix
                # If index in results we had a failure
                if idx in results:
                    results[index_of_kind] = results.pop(idx)
                else:
                    try:
                        result = check.compute(context, dataset_kind=dataset_kind)
                        context.finalize_check_result(result, check)
                        # Update header with dataset type only if both train and test ran
                        if run_train_test_checks:
                            result.header = result.get_header() + type_suffix
                        results[index_of_kind] = result
                    except Exception as exp:
                        results[index_of_kind] = CheckFailure(check, exp, type_suffix)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
