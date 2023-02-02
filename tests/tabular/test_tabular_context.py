# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
from deepchecks.tabular import Context

def test_task_type_same_with_model_or_y_pred(diabetes_split_dataset_and_model):
    # Arrange
    train, _, model = diabetes_split_dataset_and_model
    # Act
    ctx1 = Context(train, model=model)
    ctx2 = Context(train, y_pred_train=model.predict(train.features_columns))
    # Assert
    assert ctx1.task_type == ctx2.task_type
