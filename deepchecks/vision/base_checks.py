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
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
from ignite.metrics import Metric
from torch import nn

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind, ModelOnlyBaseCheck, SingleDatasetBaseCheck, TrainTestBaseCheck
from deepchecks.utils.ipython import ProgressBarGroup
from deepchecks.vision import deprecation_warnings  # pylint: disable=unused-import # noqa: F401
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.context import Context
from deepchecks.vision.utils.vision_properties import STATIC_PROPERTIES_FORMAT
from deepchecks.vision.vision_data import VisionData

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
        model: Optional[nn.Module] = None,
        model_name: str = '',
        scorers: Optional[Mapping[str, Metric]] = None,
        scorers_per_class: Optional[Mapping[str, Metric]] = None,
        device: Union[str, torch.device, None] = None,
        random_state: int = 42,
        n_samples: Optional[int] = 10_000,
        with_display: bool = True,
        train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        train_properties: Optional[STATIC_PROPERTIES_FORMAT] = None,
        test_properties: Optional[STATIC_PROPERTIES_FORMAT] = None
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        dataset: VisionData
            VisionData object to process
        model: Optional[nn.Module] , default None
            pytorch neural network module instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:

            with progressbar_factory.create_dummy(name='Validating Input'):
                # Context is copying the data object, then not using the original after the init
                context: Context = self.context_type(
                    dataset,
                    model=model,
                    model_name=model_name,
                    scorers=scorers,
                    scorers_per_class=scorers_per_class,
                    device=device,
                    random_state=random_state,
                    n_samples=n_samples,
                    with_display=with_display,
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    train_properties=train_properties,
                    test_properties=test_properties
                )
                self.initialize_run(context, DatasetKind.TRAIN)

            context.train.init_cache()

            for i, batch in enumerate(progressbar_factory.create(
                iterable=context.train,
                name='Ingesting Batches',
                unit='Batch'
            )):
                batch = Batch(batch, context, DatasetKind.TRAIN, i)
                context.train.update_cache(batch)
                self.update(context, batch, DatasetKind.TRAIN)

            with progressbar_factory.create_dummy(name='Computing Check', unit='Check'):
                result = self.compute(context, DatasetKind.TRAIN)
                context.finalize_check_result(result, self)
                context.add_is_sampled_footnote(result, DatasetKind.TRAIN)
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
        model: Optional[nn.Module] = None,
        model_name: str = '',
        scorers: Optional[Mapping[str, Metric]] = None,
        scorers_per_class: Optional[Mapping[str, Metric]] = None,
        device: Union[str, torch.device, None] = None,
        random_state: int = 42,
        n_samples: Optional[int] = 10_000,
        with_display: bool = True,
        train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        train_properties: Optional[STATIC_PROPERTIES_FORMAT] = None,
        test_properties: Optional[STATIC_PROPERTIES_FORMAT] = None
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        train_dataset: VisionData
            VisionData object, representing data an neural network was fitted on
        test_dataset: VisionData
            VisionData object, representing data an neural network predicts on
        model: Optional[nn.Module] , default None
            pytorch neural network module instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:

            with progressbar_factory.create_dummy(name='Validating Input'):
                # Context is copying the data object, then not using the original after the init
                context: Context = self.context_type(
                    train_dataset,
                    test_dataset,
                    model=model,
                    model_name=model_name,
                    scorers=scorers,
                    scorers_per_class=scorers_per_class,
                    device=device,
                    random_state=random_state,
                    n_samples=n_samples,
                    with_display=with_display,
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    train_properties=train_properties,
                    test_properties=test_properties
                )
                self.initialize_run(context)

            train_pbar = progressbar_factory.create(
                iterable=context.train,
                name='Ingesting Batches - Train Dataset',
                unit='Batch'
            )

            context.train.init_cache()

            for i, batch in enumerate(train_pbar):
                batch = Batch(batch, context, DatasetKind.TRAIN, i)
                context.train.update_cache(batch)
                self.update(context, batch, DatasetKind.TRAIN)

            context.test.init_cache()

            for i, batch in enumerate(progressbar_factory.create(
                iterable=context.test,
                name='Ingesting Batches - Test Dataset',
                unit='Batch'
            )):
                batch = Batch(batch, context, DatasetKind.TEST, i)
                context.test.update_cache(batch)
                self.update(context, batch, DatasetKind.TEST)

            with progressbar_factory.create_dummy(name='Computing Check', unit='Check'):
                result = self.compute(context)
                context.finalize_check_result(result, self)
                context.add_is_sampled_footnote(result)
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
        model_name: str = '',
        scorers: Optional[Mapping[str, Metric]] = None,
        scorers_per_class: Optional[Mapping[str, Metric]] = None,
        device: Union[str, torch.device, None] = None,
        random_state: int = 42,
        n_samples: Optional[int] = None,
        with_display: bool = True,
        train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] = None,
        train_properties: Optional[STATIC_PROPERTIES_FORMAT] = None,
        test_properties: Optional[STATIC_PROPERTIES_FORMAT] = None
    ) -> CheckResult:
        """Run check.

        Parameters
        ----------
        model: nn.Module
            pytorch neural network module instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None

        with ProgressBarGroup() as progressbar_factory:

            with progressbar_factory.create_dummy(name='Validating Input'):
                context: Context = self.context_type(
                    model=model,
                    model_name=model_name,
                    scorers=scorers,
                    scorers_per_class=scorers_per_class,
                    device=device,
                    random_state=random_state,
                    n_samples=n_samples,
                    with_display=with_display,
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    train_properties=train_properties,
                    test_properties=test_properties
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
