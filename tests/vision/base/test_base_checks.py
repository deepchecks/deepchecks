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

from hamcrest import assert_that, calling, is_, raises

from deepchecks import CheckResult
from deepchecks.core import DatasetKind
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.datasets.detection import coco


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
    # Act Assert
    assert_that(SingleDatasetCheck().initialize_run(None, None), is_(None))
    assert_that(TrainTestCheck().initialize_run(None), is_(None))
    assert_that(ModelOnlyCheck().initialize_run(None), is_(None))
