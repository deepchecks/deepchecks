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
"""Module for vision base checks."""
from typing import Any, Optional

from torch import nn

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind, ModelOnlyBaseCheck, SingleDatasetBaseCheck, TrainTestBaseCheck
from deepchecks.utils.ipython import ProgressBarGroup
from deepchecks.vision import deprecation_warnings  # pylint: disable=unused-import # noqa: F401
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
]


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    @docstrings
    def run(
            self,
            dataset: VisionData,
            random_state: int = 42,
            with_display: bool = True,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        dataset: VisionData
            VisionData object to process
        {additional_run_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:
            with progressbar_factory.create_dummy(name='Validating Input'):
                # Context is copying the data object, then not using the original after the init
                context: Context = self.context_type(
                    dataset,
                    random_state=random_state,
                    with_display=with_display,
                )
                self.initialize_run(context, DatasetKind.TRAIN)

            dataset.init_cache()

            for i, batch in enumerate(progressbar_factory.create(
                    iterable=dataset,
                    name='Ingesting Batches',
                    unit='Batch'
            )):
                batch = BatchWrapper(batch, dataset)
                dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                self.update(context, batch, DatasetKind.TRAIN)

            with progressbar_factory.create_dummy(name='Computing Check', unit='Check'):
                result = self.compute(context, DatasetKind.TRAIN)
                context.finalize_check_result(result, self, DatasetKind.TRAIN)
                return result

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

    @docstrings
    def run(
            self,
            train_dataset: VisionData,
            test_dataset: VisionData,
            random_state: int = 42,
            with_display: bool = True,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        train_dataset : VisionData
            VisionData object, representing data the model was fitted on
        test_dataset : VisionData
            VisionData object, representing data the models predicts on
        {additional_run_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:

            with progressbar_factory.create_dummy(name='Validating Input'):
                # Context is copying the data object, then not using the original after the init
                context: Context = self.context_type(
                    train_dataset,
                    test_dataset,
                    random_state=random_state,
                    with_display=with_display,
                )
                self.initialize_run(context)

            train_pbar = progressbar_factory.create(
                iterable=train_dataset,
                name='Ingesting Batches - Train Dataset',
                unit='Batch'
            )
            train_dataset.init_cache()
            for i, batch in enumerate(train_pbar):
                batch = BatchWrapper(batch, train_dataset)
                train_dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                self.update(context, batch, DatasetKind.TRAIN)

            test_dataset.init_cache()
            for i, batch in enumerate(progressbar_factory.create(
                    iterable=test_dataset,
                    name='Ingesting Batches - Test Dataset',
                    unit='Batch'
            )):
                batch = BatchWrapper(batch, test_dataset)
                test_dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                self.update(context, batch, DatasetKind.TEST)

            with progressbar_factory.create_dummy(name='Computing Check', unit='Check'):
                result = self.compute(context)
                context.finalize_check_result(result, self)
                return result

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

    @docstrings
    def run(
            self,
            model: nn.Module,
            random_state: int = 42,
            with_display: bool = True,
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        {additional_run_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:
            with progressbar_factory.create_dummy(name='Validating Input'):
                context: Context = self.context_type(  # currently no model is passed to context
                    random_state=random_state,
                    with_display=with_display,
                )
                self.initialize_run(context)

            with progressbar_factory.create_dummy(name='Computing Check', unit='Check'):
                result = self.compute(context)
                context.finalize_check_result(result, self)
                return result

    def initialize_run(self, context: Context):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def compute(self, context: Context) -> CheckResult:
        """Compute final check result."""
        raise NotImplementedError()
