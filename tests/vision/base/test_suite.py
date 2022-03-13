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