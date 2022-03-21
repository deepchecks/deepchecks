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
import typing as t
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from hamcrest import (
    assert_that,
    calling,
    raises,
    equal_to,
    has_entries,
    instance_of,
    all_of,
    is_,
)

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError, DatasetValidationError, DeepchecksNotSupportedError
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.base_checks import SingleDatasetCheck, TrainTestCheck, ModelOnlyCheck
from deepchecks.vision.suite import Suite
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist


def test_suite_execution():
    coco_dataset = coco.load_dataset(object_type='VisionData')
    executions = defaultdict(int)

    class DummyCheck(SingleDatasetCheck):
        def initialize_run(self, context, dataset_kind: DatasetKind):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context, dataset_kind: DatasetKind) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(0)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(1)

    suite = Suite("test",
                           DummyCheck(),
                           DummyTrainTestCheck())
    result = suite.run(train_dataset=coco_dataset, test_dataset=coco_dataset)

    assert_that(result.results[0].value, is_(0))
    assert_that(result.results[2].value, is_(1))
    assert_that(executions, is_({'initialize_run': 3, 'update': 8, 'compute': 3}))


def test_suite_execution_with_initalize_exeption():
    coco_dataset = coco.load_dataset(object_type='VisionData')
    executions = defaultdict(int)

    class DummyCheck(SingleDatasetCheck):
        def initialize_run(self, context, dataset_kind: DatasetKind):
            executions["initialize_run"] += 1
            raise DeepchecksValueError('bad init')

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context, dataset_kind: DatasetKind) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1
            raise DeepchecksValueError('bad init')

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    suite = Suite("test",
                  DummyCheck(),
                  DummyTrainTestCheck())
    result = suite.run(train_dataset=coco_dataset, test_dataset=coco_dataset)

    assert_that(result.results[0].exception, instance_of(DeepchecksValueError))
    assert_that(result.results[0].exception.message, is_('bad init'))

    assert_that(result.results[1].exception, instance_of(DeepchecksValueError))
    assert_that(result.results[1].exception.message, is_('bad init'))

    assert_that(executions, is_({'initialize_run': 3}))


def test_suite_execution_with_exception_on_compute():
    coco_dataset = coco.load_dataset(object_type='VisionData')
    executions = defaultdict(int)

    class DummyCheck(SingleDatasetCheck):
        def initialize_run(self, context, dataset_kind: DatasetKind):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context, dataset_kind: DatasetKind) -> CheckResult:
            executions["compute"] += 1
            raise DeepchecksValueError('bad compute')
            return CheckResult(None)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            raise DeepchecksValueError('bad compute')

    suite = Suite("test",
                  DummyCheck(),
                  DummyTrainTestCheck())
    result = suite.run(train_dataset=coco_dataset, test_dataset=coco_dataset)

    assert_that(result.results[0].exception, instance_of(DeepchecksValueError))
    assert_that(result.results[0].exception.message, is_('bad compute'))

    assert_that(result.results[1].exception, instance_of(DeepchecksValueError))
    assert_that(result.results[1].exception.message, is_('bad compute'))

    assert_that(executions, is_({'initialize_run': 3, 'update': 8, 'compute': 3}))


def test_suite_execution_with_missing_train():
    coco_dataset = coco.load_dataset(object_type='VisionData')
    executions = defaultdict(int)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            raise DeepchecksValueError('bad compute')

    suite = Suite("test",
                  DummyTrainTestCheck())
    assert_that(calling(suite.run).with_args(test_dataset=coco_dataset),
                raises(DatasetValidationError,
                       "Can't initialize context with only test. if you have single dataset, initialize it as train"))

    assert_that(executions, is_({}))


def test_suite_execution_with_missing_test():
    coco_dataset = coco.load_dataset(object_type='VisionData')
    executions = defaultdict(int)

    class DummyTrainTestCheck(TrainTestCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            executions["update"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    suite = Suite("test",
                  DummyTrainTestCheck())
    r = suite.run(train_dataset=coco_dataset)

    assert_that(r.results[0].exception, instance_of(DeepchecksNotSupportedError))
    assert_that(r.results[0].exception.message,
                is_('Check is irrelevant if not supplied with both train and test datasets'))

    assert_that(executions, is_({'initialize_run': 1}))


def test_suite_model_only_check():
    executions = defaultdict(int)

    class DummyModelOnlyCheck(ModelOnlyCheck):
        def initialize_run(self, context):
            executions["initialize_run"] += 1

        def compute(self, context) -> CheckResult:
            executions["compute"] += 1
            return CheckResult(None)

    suite = Suite("test",
                  DummyModelOnlyCheck())
    suite.run(model="some model")

    assert_that(executions, is_({'initialize_run': 1, 'compute': 1}))
