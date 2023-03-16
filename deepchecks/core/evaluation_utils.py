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
# pylint: disable=import-outside-toplevel
"""Module containing the fix classes and methods."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

__all__ = [
    'evaluate_change_in_performance',
]


# This function is in core and not tabular because widgets are in Core, but it expects tabular Dataset object
def evaluate_change_in_performance(old_train_ds, old_test_ds, new_train_ds, new_test_ds, task_type, random_state=42):
    cat_features = [old_train_ds.features.index(col) for col in old_train_ds.cat_features]
    task_type = task_type.value

    if task_type == 'regression':
        old_model = CatBoostRegressor(cat_features=cat_features, random_state=random_state)
        new_model = CatBoostRegressor(cat_features=cat_features, random_state=random_state)

        old_model.fit(old_train_ds.features_columns, old_train_ds.label_col, verbose=0)
        new_model.fit(new_train_ds.features_columns, new_train_ds.label_col, verbose=0)
    else:
        old_model = CatBoostClassifier(cat_features=cat_features, random_state=random_state)
        new_model = CatBoostClassifier(cat_features=cat_features, random_state=random_state)
        old_model.fit(old_train_ds.features_columns, old_train_ds.label_col, verbose=0)
        new_model.fit(new_train_ds.features_columns, new_train_ds.label_col, verbose=0)
    old_score = old_model.score(old_test_ds.features_columns, old_test_ds.label_col)
    new_score = new_model.score(new_test_ds.features_columns, new_test_ds.label_col)
    return (new_score-old_score) / old_score




