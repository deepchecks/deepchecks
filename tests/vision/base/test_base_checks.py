# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
#
from collections import defaultdict

import torch
from torch import nn
from hamcrest import (
    assert_that,
    calling,
    raises,
    equal_to,
    has_properties,
    has_property,
    instance_of,
    same_instance,
    all_of,
    is_
)

from deepchecks import CheckResult
from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksValueError, ValidationError, ModelValidationError
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.errors import DatasetValidationError
from deepchecks.vision.base_checks import SingleDatasetCheck, TrainTestCheck, ModelOnlyCheck
from deepchecks.vision import ClassificationData, DetectionData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.vision_data import TaskType, VisionData


def test_run_base_checks():
    # Arrange
    coco_dataset = coco.load_dataset(object_type='VisionData')
    coco_model = coco.load_model()
    executions = defaultdict(int)

    class DummyCheck(SingleDatasetCheck):
        def initialize_run(self, context, dataset_kind: DatasetKind):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context, dataset_kind: DatasetKind) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    class DummyModelCheck(ModelOnlyCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    # Act
    DummyCheck().run(coco_dataset)
    DummyTrainTestCheck().run(coco_dataset, coco_dataset)
    DummyModelCheck().run(coco_model)

    # Assert
    assert_that(executions, is_({"initialize_run": 3, "compute": 3, "update": 6}))


def test_base_check_raise_not_implemented():
    context = None
    batch = None
    dataset_kind = DatasetKind.TRAIN

    # Act Assert
    assert_that(calling(SingleDatasetCheck().update).with_args(context, batch, dataset_kind),
                raises(NotImplementedError))
    assert_that(calling(TrainTestCheck().update).with_args(context, batch, dataset_kind),
                raises(NotImplementedError))

    assert_that(calling(SingleDatasetCheck().compute).with_args(context, dataset_kind),
                raises(NotImplementedError))
    assert_that(calling(TrainTestCheck().compute).with_args(context),
                raises(NotImplementedError))
    assert_that(calling(ModelOnlyCheck().compute).with_args(context),
                raises(NotImplementedError))


def test_initialize_run():
    assert_that(SingleDatasetCheck().initialize_run(None, None), is_(None))
    assert_that(ModelOnlyCheck().initialize_run(None), is_(None))
