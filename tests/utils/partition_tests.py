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
"""Test partitions utils"""
import pandas as pd
from hamcrest import assert_that, equal_to
from sklearn.tree import DecisionTreeRegressor

from deepchecks.utils.performance.partition import (DeepchecksFilter, convert_tree_leaves_into_filters,
                                                    intersect_two_filters)


def test_iris_tree_to_filters(iris_dataset):
    model = DecisionTreeRegressor(max_depth=2)
    data = iris_dataset.features_columns.iloc[:, 2:]
    model.fit(data, iris_dataset.label_col)
    leaves = convert_tree_leaves_into_filters(model.tree_, list(data.columns))
    leaf0_samples = leaves[0].filter(data)
    assert_that(len(leaf0_samples), equal_to(50))
    leaf1_samples = leaves[1].filter(data)
    assert_that(len(leaf1_samples), equal_to(54))
    leaf2_samples = leaves[2].filter(data)
    assert_that(len(leaf2_samples), equal_to(46))

    leaves_combined = pd.concat([leaf0_samples, leaf1_samples, leaf2_samples]).drop_duplicates()
    assert_that(len(leaves_combined), equal_to(len(data.drop_duplicates())))


def test_merge_filters(iris_clean):
    filter1 = DeepchecksFilter([lambda df, a=2: df['petal length (cm)'] <= a])
    filter2 = DeepchecksFilter([lambda df, a=1.5: df['petal length (cm)'] <= a])
    filter3 = intersect_two_filters(filter1, filter2)

    filter2_data = filter2.filter(iris_clean.data)
    filter3_data = filter3.filter(iris_clean.data)

    assert_that(len(filter2_data), equal_to(len(filter3_data)))
    assert_that(list(filter2_data.iloc[0, :]), equal_to(list(filter3_data.iloc[0, :])))


def test_empty_merge_filters(iris_clean):
    filter1 = DeepchecksFilter([lambda df, a=2: df['petal length (cm)'] > a])
    filter2 = DeepchecksFilter([lambda df, a=1.5: df['petal length (cm)'] <= a])
    filter3 = intersect_two_filters(filter1, filter2)

    filter3_data = filter3.filter(iris_clean.data)
    assert_that(len(filter3_data), equal_to(0))


def test_no_effect_merge_filters(iris_clean):
    filter1 = DeepchecksFilter([lambda df, a=2: df['petal length (cm)'] > a])
    filter2 = DeepchecksFilter()
    filter3 = intersect_two_filters(filter1, filter2)

    filter1_data = filter1.filter(iris_clean.data)
    filter2_data = filter2.filter(iris_clean.data)
    filter3_data = filter3.filter(iris_clean.data)
    assert_that(len(filter3_data), equal_to(len(filter1_data)))
    assert_that(len(filter2_data), equal_to(len(iris_clean.data)))

