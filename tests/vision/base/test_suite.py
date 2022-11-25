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

from hamcrest import assert_that, calling, contains_exactly, has_length, instance_of, is_, raises

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.suite import Suite
from deepchecks.vision.suites.default_suites import full_suite
from tests.conftest import get_expected_results_length, validate_suite_result


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
    args = {'train_dataset': coco_dataset, 'test_dataset': coco_dataset}
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)

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

    args = {'train_dataset': coco_dataset, 'test_dataset': coco_dataset}
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)

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

    args = {'train_dataset': coco_dataset, 'test_dataset': coco_dataset}
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)

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

    args = {'train_dataset': coco_dataset}
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)

    assert_that(result.results[0].exception, instance_of(DeepchecksNotSupportedError))
    assert_that(result.results[0].exception.message,
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

    args = {'model': "some model"}
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)

    assert_that(executions, is_({'initialize_run': 1, 'compute': 1}))


def test_full_suite_execution_mnist(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
    arguments = (
        dict(train_dataset=mnist_dataset_train, test_dataset=mnist_dataset_test,
             model=mock_trained_mnist, device=device),
        dict(train_dataset=mnist_dataset_train,
             model=mock_trained_mnist, device=device),
        dict(train_dataset=mnist_dataset_train,
             model=mock_trained_mnist, device=device, with_display=False),
    )

    for args in arguments:
        result = suite.run(**args)
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def test_full_suite_execution_coco(coco_train_visiondata, coco_test_visiondata,
                                   mock_trained_yolov5_object_detection, device):
    suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
    arguments = (
        dict(train_dataset=coco_train_visiondata, test_dataset=coco_test_visiondata,
             model=mock_trained_yolov5_object_detection, device=device),
        dict(train_dataset=coco_train_visiondata,
             model=mock_trained_yolov5_object_detection, device=device),
        dict(train_dataset=coco_train_visiondata,
             model=mock_trained_yolov5_object_detection, device=device, with_display=False),
    )

    for args in arguments:
        result = suite.run(**args)
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def test_single_dataset(coco_train_visiondata, coco_test_visiondata,
                        mock_trained_yolov5_object_detection, device):
    suite = full_suite()
    res_train = suite.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection,
                          device=device, n_samples=100, with_display=False, run_single_dataset='Train')
    expected_train_headers = ['Class Performance',
                              'Train Test Label Drift',
                              'Feature Label Correlation Change',
                              'Train Test Prediction Drift',
                              'Image Segment Performance - Train Dataset',
                              'New Labels',
                              'Mean Average Precision Report - Train Dataset',
                              'Image Property Drift',
                              'Image Dataset Drift',
                              'Image Property Outliers - Train Dataset',
                              'Label Property Outliers - Train Dataset',
                              'Property Label Correlation - Train Dataset',
                              'Mean Average Recall Report - Train Dataset',
                              'Confusion Matrix - Train Dataset',
                              'Heatmap Comparison',
                              'Simple Model Comparison',
                              'Model Error Analysis']

    res_test = suite.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection,
                         device=device, n_samples=100, with_display=False, run_single_dataset='Test')
    res_full = suite.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection,
                         device=device, n_samples=100, with_display=False)
    res_names = [x.get_header() for x in res_train.results]
    assert_that(res_names, contains_exactly(*expected_train_headers))
    assert_that(res_test.results, has_length(17))
    assert_that(res_full.results, has_length(24))
