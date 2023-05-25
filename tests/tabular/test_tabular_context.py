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
import pandas as pd
from hamcrest import assert_that, calling, raises

from deepchecks import Dataset
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular import Context


def test_task_type_same_with_model_or_y_pred(diabetes_split_dataset_and_model):
    # Arrange
    train, _, model = diabetes_split_dataset_and_model
    # Act
    ctx1 = Context(train, model=model)
    ctx2 = Context(train, y_pred_train=model.predict(train.features_columns))
    # Assert
    assert ctx1.task_type == ctx2.task_type


def test_dataset_bad_cat_features_param():
    # Arrange
    df = pd.DataFrame(
        {
            'binary_feature': [0, 1, 1, 0, 0, 1],
            'string_feature': ['ahhh', 'no', 'weeee', 'arg', 'eh', 'E'],
            'numeric_label': [3, 1, 5, 2, 1, 1],
        })

    # Act & Assert
    assert_that(calling(Dataset).with_args(df, label='numeric_label', cat_features=['binary_feature']),
                raises(DeepchecksValueError, 'Feature string_feature passed as numeric feature '
                                             'but cannot be cast to float.'))

