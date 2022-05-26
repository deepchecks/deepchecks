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
import copy

import torch
from hamcrest import (assert_that, calling, close_to, equal_to,
                      has_properties, has_property, instance_of, is_, raises)
from deepchecks.core.check_result import CheckResult

from deepchecks.vision.base_checks import SingleDatasetCheck
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.checks.model_evaluation.class_performance import ClassPerformance
from deepchecks.vision.context import Context
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.vision_data import VisionData


class _StaticPred(SingleDatasetCheck):
    def initialize_run(self, context: Context, dataset_kind):
        self._pred_index = {}

    def update(self, context: Context, batch: Batch, dataset_kind):
        predictions = batch.predictions
        indexes = [batch.get_index_in_dataset(index) for index in range(len(predictions))]
        self._pred_index.update(dict(zip(indexes, predictions)))

    def compute(self, context: Context, dataset_kind) -> CheckResult:
        sorted_values = [v for _, v in sorted(self._pred_index.items(), key=lambda item: item[0])]
        if context.get_data_by_kind(dataset_kind).task_type == TaskType.CLASSIFICATION:
            sorted_values = torch.stack(sorted_values)
        return CheckResult(sorted_values)


def _create_static_predictions(train: VisionData, test: VisionData, model):
    static_preds = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_pred = _StaticPred().run(vision_data, model=model, n_samples=None).value
        else:
            static_pred = None
        static_preds.append(static_pred)
    train_preds, tests_preds = static_preds
    return train_preds, tests_preds


# copied from class_performance_test
def test_class_performance_mnist_largest(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    train_preds, tests_preds = _create_static_predictions(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist)
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test,
                       train_predictions=train_preds, test_predictions=tests_preds,
                       device=device, n_samples=None)
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(set(result.value['Class'])), equal_to(2))
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.977, 0.001))
    assert_that(first_row['Number of samples'], equal_to(6742))
    assert_that(first_row['Class'], equal_to(1))