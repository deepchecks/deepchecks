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
from typing import Tuple, Mapping, Optional, Union, Dict, List
from collections import OrderedDict

import torch
from torch import nn
from ignite.metrics import Metric

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import DatasetKind
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.batch_wrapper import Batch


__all__ = ['Suite']


logger = logging.getLogger('deepchecks')


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
            random_state: int = 42,
            n_samples: Optional[int] = 10_000,
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
        n_samples : int, default: 10,000
            number of samples to draw from the dataset.
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        all_pbars = []
        progress_bar = ProgressBar('Validating Input', 1, unit='')
        all_pbars.append(progress_bar)
        context = Context(
            train_dataset,
            test_dataset,
            model,
            scorers=scorers,
            scorers_per_class=scorers_per_class,
            device=device,
            random_state=random_state,
            n_samples=n_samples
        )
        progress_bar.inc_progress()

        results: Dict[
            Union[str, int],
            Union[CheckResult, CheckFailure]
        ] = OrderedDict({})

        run_train_test_checks = train_dataset is not None and test_dataset is not None
        non_single_checks = {k: check for k, check in self.checks.items() if not isinstance(check, SingleDatasetCheck)}

        # Initialize here all the checks that are not single dataset, since those are initialized inside the update loop
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
                progress_bars=all_pbars
            )

        if test_dataset is not None:
            self._update_loop(
                context=context,
                run_train_test_checks=run_train_test_checks,
                results=results,
                dataset_kind=DatasetKind.TEST,
                progress_bars=all_pbars
            )

        # Need to compute only on not SingleDatasetCheck, since they computed inside the loop
        if non_single_checks:
            progress_bar = ProgressBar('Computing Checks', len(non_single_checks), unit='Check')
            all_pbars.append(progress_bar)
            for check_idx, check in non_single_checks.items():
                progress_bar.set_text(check.name())
                try:
                    # if check index in results we had failure
                    if check_idx not in results:
                        result = check.finalize_check_result(check.compute(context))
                        results[check_idx] = result
                except Exception as exp:
                    results[check_idx] = CheckFailure(check, exp)
                progress_bar.inc_progress()

        # The results are ordered as they ran instead of in the order they were defined, therefore sort by key
        sorted_result_values = [value for name, value in sorted(results.items(), key=lambda pair: str(pair[0]))]

        # Close all progress bars
        for pbar in all_pbars:
            pbar.close()

        footnote = context.get_is_sampled_footnote()
        extra_info = [footnote] if footnote else []
        return SuiteResult(self.name, sorted_result_values, extra_info)

    def _update_loop(
        self,
        context: Context,
        run_train_test_checks: bool,
        results: Dict[Union[str, int], Union[CheckResult, CheckFailure]],
        dataset_kind: DatasetKind,
        progress_bars: List
    ):
        type_suffix = ' - Test Dataset' if dataset_kind == DatasetKind.TEST else ' - Train Dataset'
        vision_data = context.get_data_by_kind(dataset_kind)
        n_batches = len(vision_data)
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

        progress_bar = ProgressBar('Ingesting Batches' + type_suffix, n_batches, unit='Batch')
        progress_bars.append(progress_bar)

        # Run on all the batches
        batch_start_index = 0
        for batch in vision_data:
            batch = Batch(batch, context, dataset_kind, batch_start_index)
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

            batch_start_index += len(batch)
            progress_bar.inc_progress()

        # SingleDatasetChecks have different handling. If we had failure in them need to add suffix to the index of
        # the results, else need to compute it.
        if single_dataset_checks:
            progress_bar = ProgressBar('Computing Single Dataset Checks' + type_suffix,
                                       len(single_dataset_checks),
                                       unit='Check')
            progress_bars.append(progress_bar)
            for idx, check in single_dataset_checks.items():
                progress_bar.set_text(check.name())
                index_of_kind = str(idx) + type_suffix
                # If index in results we had a failure
                if idx in results:
                    results[index_of_kind] = results.pop(idx)
                else:
                    try:
                        result = check.compute(context, dataset_kind=dataset_kind)
                        result = check.finalize_check_result(result)
                        # Update header with dataset type only if both train and test ran
                        if run_train_test_checks:
                            result.header = result.get_header() + type_suffix
                        results[index_of_kind] = result
                    except Exception as exp:
                        results[index_of_kind] = CheckFailure(check, exp, type_suffix)
                progress_bar.inc_progress()

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
