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
import logging
from typing import Optional, Any, Union

import torch
from torch import nn

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import (
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck,
    DatasetKind
)
from deepchecks.vision.context import Context
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.batch_wrapper import Batch


logger = logging.getLogger('deepchecks')


__all__ = [
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
]


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
        # Context is copying the data object, then not using the original after the init
        context: Context = self.context_type(dataset,
                                             model=model,
                                             device=device,
                                             random_state=random_state)

        self.initialize_run(context, DatasetKind.TRAIN)

        context.train.init_cache()
        batch_start_index = 0
        for batch in context.train:
            batch = Batch(batch, context, DatasetKind.TRAIN, batch_start_index)
            context.train.update_cache(batch)
            self.update(context, batch, DatasetKind.TRAIN)
            batch_start_index += len(batch)

        return self.finalize_check_result(self.compute(context, DatasetKind.TRAIN))

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
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
        # Context is copying the data object, then not using the original after the init
        context: Context = self.context_type(train_dataset,
                                             test_dataset,
                                             model=model,
                                             device=device,
                                             random_state=random_state)

        self.initialize_run(context)

        context.train.init_cache()
        batch_start_index = 0
        for batch in context.train:
            batch = Batch(batch, context, DatasetKind.TRAIN, batch_start_index)
            context.train.update_cache(batch)
            self.update(context, batch, DatasetKind.TRAIN)
            batch_start_index += len(batch)

        context.test.init_cache()
        batch_start_index = 0
        for batch in context.test:
            batch = Batch(batch, context, DatasetKind.TEST, batch_start_index)
            context.test.update_cache(batch)
            self.update(context, batch, DatasetKind.TEST)
            batch_start_index += len(batch)

        return self.finalize_check_result(self.compute(context))

    def initialize_run(self, context: Context):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
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
        return self.finalize_check_result(self.compute(context))

    def initialize_run(self, context: Context):
        """Initialize run before starting updating on batches. Optional."""
        pass

    def compute(self, context: Context) -> CheckResult:
        """Compute final check result."""
        raise NotImplementedError()
