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
from copy import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np

from deepchecks.core.check_result import BaseCheckResult, CheckFailure
from deepchecks.core.checks import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.utils.ipython import ProgressBarGroup
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

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
            random_state: int = 42,
            with_display: bool = True,
            max_samples: Optional[int] = None,
            run_single_dataset: Optional[str] = None
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset : Optional[VisionData] , default: None
            VisionData object, representing data the model was fitted on
        test_dataset : Optional[VisionData] , default: None
            VisionData object, representing data the models predicts on
        {additional_run_params:2*indent}
        max_samples : Optional[int] , default: None
            Each check will run on a number of samples which is the minimum between the n_samples parameter of the check
            and this parameter. If this argument is None then the number of samples for each check will be
            determined by the n_samples argument.
        run_single_dataset: Optional[str], default None
            'Train', 'Test' , or None to run on both train and test.

        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        single_dataset_checks_train = {str(k) + ' - Train Dataset': copy(check) for k, check in self.checks.items() if
                                       isinstance(check, SingleDatasetCheck)} if run_single_dataset != 'Test' else {}
        single_dataset_checks_test = {str(k) + ' - Test Dataset': copy(check) for k, check in self.checks.items() if
                                      isinstance(check, SingleDatasetCheck)} if run_single_dataset != 'Train' else {}
        train_test_checks = {str(k): check for k, check in self.checks.items() if isinstance(check, TrainTestCheck)}

        results: Dict[Union[str, int], BaseCheckResult] = OrderedDict({})
        max_samples = max_samples or np.inf

        with ProgressBarGroup() as progressbar_factory:
            context = Context(train_dataset, test_dataset,
                              random_state=random_state, with_display=with_display)
            # Initialize train test checks
            if train_dataset is None or test_dataset is None:
                for name, check in list(train_test_checks.items()):
                    msg = 'Check is irrelevant if not supplied with both train and test datasets'
                    results[name] = self._get_unsupported_failure(check, msg)
                train_test_checks = {}
            for name, check in copy(train_test_checks).items():
                try:
                    check.initialize_run(context)
                except Exception as exp:
                    results[name] = CheckFailure(check, exp)
                    train_test_checks.pop(name)

            if train_dataset is not None:
                for name, check in list(single_dataset_checks_train.items()):
                    try:
                        check.initialize_run(context, dataset_kind=DatasetKind.TRAIN)
                    except Exception as exp:
                        results[name] = CheckFailure(check, exp)
                        single_dataset_checks_train.pop(name)
                self._update_loop(context=context, train_test_checks=train_test_checks,
                                  single_dataset_checks=single_dataset_checks_train, results=results,
                                  dataset_kind=DatasetKind.TRAIN, progressbar_factory=progressbar_factory,
                                  max_samples=max_samples)

            if test_dataset is not None:
                for name, check in list(single_dataset_checks_test.items()):
                    try:
                        check.initialize_run(context, dataset_kind=DatasetKind.TRAIN)
                    except Exception as exp:
                        results[name] = CheckFailure(check, exp)
                        single_dataset_checks_test.pop(name)
                self._update_loop(context=context, train_test_checks=train_test_checks,
                                  single_dataset_checks=single_dataset_checks_test, results=results,
                                  dataset_kind=DatasetKind.TEST, progressbar_factory=progressbar_factory,
                                  max_samples=max_samples)

            # Need to compute only on not SingleDatasetCheck, since they computed inside the loop
            progress_bar = progressbar_factory.create(iterable=list(train_test_checks.items()), unit='Check',
                                                      name='Computing Train Test Checks')
            for name, check in progress_bar:
                progress_bar.set_postfix({'Check': check.name()})
                try:
                    result = check.compute(context)
                    context.finalize_check_result(result, check)
                    results[name] = result
                except Exception as exp:
                    results[name] = CheckFailure(check, exp)

        sorted_result_values = [value for name, value in sorted(results.items(), key=lambda pair: str(pair[0]))]
        return SuiteResult(self.name, sorted_result_values)

    @classmethod
    def _update_loop(cls, context: Context, dataset_kind: DatasetKind, results: Dict[Union[str, int], BaseCheckResult],
                     progressbar_factory: ProgressBarGroup, train_test_checks, single_dataset_checks, max_samples):
        checks_to_update = {**train_test_checks, **single_dataset_checks}
        vision_data = context.get_data_by_kind(dataset_kind)

        # Update loop over the batches
        with progressbar_factory.create_dummy(name='Processing Batches:' + vision_data.name):
            for batch in vision_data:
                batch = BatchWrapper(batch, vision_data.task_type, vision_data.number_of_images_cached)
                vision_data.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                for name, check in list(checks_to_update.items()):
                    try:
                        check.update(context, batch, dataset_kind=dataset_kind)
                        if vision_data.number_of_images_cached > np.min((max_samples, check.n_samples or np.inf)):
                            checks_to_update.pop(name)
                    except Exception as exp:
                        results[name] = CheckFailure(check, exp, vision_data.name)
                        checks_to_update.pop(name)
                        if name in single_dataset_checks:
                            single_dataset_checks.pop(name)
                        else:
                            train_test_checks.pop(name)
                if len(checks_to_update) == 0:
                    break

        # Compute for single dataset checks
        checks_pbar = progressbar_factory.create(iterable=list(single_dataset_checks.items()), unit='Check',
                                                 name='Computing Single Dataset Checks ' + vision_data.name)
        for name, check in checks_pbar:
            checks_pbar.set_postfix({'Check': check.name()}, refresh=False)
            try:
                result = check.compute(context, dataset_kind=dataset_kind)
                context.finalize_check_result(result, check, dataset_kind=dataset_kind)
                results[name] = result
            except Exception as exp:
                results[name] = CheckFailure(check, exp, vision_data.name)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))
