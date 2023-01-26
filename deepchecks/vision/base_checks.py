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
from typing import Any

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
    def run(self, dataset: VisionData, random_state: int = 42, with_display: bool = True) -> CheckResult:
        """Run check.

        Parameters
        ----------
        dataset: VisionData
            VisionData object to process
        {additional_run_params:2*indent}
        """
        with ProgressBarGroup() as progressbar_factory:
            context: Context = self.context_type(train=dataset, random_state=random_state, with_display=with_display)
            self.initialize_run(context, DatasetKind.TRAIN)

            with progressbar_factory.create_dummy(name='Processing Batches'):
                for batch in context.train:
                    batch = BatchWrapper(batch, context.train.task_type, context.train.number_of_images_cached)
                    context.train.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                    self.update(context, batch, DatasetKind.TRAIN)
                    if self.n_samples is not None and context.train.number_of_images_cached >= self.n_samples:
                        break

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
    def run(self, train_dataset: VisionData, test_dataset: VisionData,
            random_state: int = 42, with_display: bool = True) -> CheckResult:
        """Run check.

        Parameters
        ----------
        train_dataset : VisionData
            VisionData object, representing data the model was fitted on
        test_dataset : VisionData
            VisionData object, representing data the models predicts on
        {additional_run_params:2*indent}
        """
        with ProgressBarGroup() as progressbar_factory:
            context: Context = self.context_type(train=train_dataset, test=test_dataset,
                                                 random_state=random_state, with_display=with_display)
            self.initialize_run(context)

            with progressbar_factory.create_dummy(name='Processing Train Batches'):
                for batch in context.train:
                    batch = BatchWrapper(batch, context.train.task_type, context.train.number_of_images_cached)
                    context.train.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                    self.update(context, batch, DatasetKind.TRAIN)
                    if self.n_samples is not None and context.train.number_of_images_cached >= self.n_samples:
                        break

            with progressbar_factory.create_dummy(name='Processing Test Batches'):
                for batch in context.test:
                    batch = BatchWrapper(batch, context.test.task_type, context.test.number_of_images_cached)
                    context.test.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
                    self.update(context, batch, DatasetKind.TEST)
                    if self.n_samples is not None and context.test.number_of_images_cached >= self.n_samples:
                        break

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
    def run(self, model, random_state: int = 42, with_display: bool = True) -> CheckResult:
        """Run check.

        Parameters
        ----------
        model
            Model to run the check on
        {additional_run_params:2*indent}
        """
        with ProgressBarGroup() as progressbar_factory:
            # Currently we do not receive model into context since there are no model only checks
            context: Context = self.context_type(random_state=random_state, with_display=with_display)
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
