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
)

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.dataset import VisionData
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.base import SingleDatasetCheck
from deepchecks.vision.base import Suite
from deepchecks.vision.utils import ClassificationLabelFormatter
from deepchecks.vision.utils import DetectionLabelFormatter
from deepchecks.vision.utils.base_formatters import BaseLabelFormatter
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.classification import mnist



# def test_suite_execution_with_list_of_single_dataset_checks():
#     coco_dataset = coco.load_dataset(train=True, object_type="Dataset")
#     executions = {}

#     class DummyCheck(SingleDatasetCheck):

#         def initialize_run(self, context):
#             executions["initialize_run"] = executions.get("initialize_run", 0) + 1

#         def update(self, context, batch):
#             executions["update"] = executions.get("update", 0) + 1
            
#         def compute(self, context) -> CheckResult:
#             executions["compute"] = executions.get("compute", 0) + 1
#             return CheckResult(None)

    
#     suite = Suite("test", DummyCheck())
#     suite_result = suite.run(train_dataset=coco_dataset)

#     breakpoint()