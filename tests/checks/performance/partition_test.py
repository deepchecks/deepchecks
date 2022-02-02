# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for partition columns function."""
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.utils.performance.partition import partition_column
import hamcrest as h


def test_column_partition_numerical(diabetes):
    # Arrange
    train, _ = diabetes
    # Act
    filters = partition_column(train, 'age', 5)
    filter_results = [{'count': v.filter(train.data).size, 'label': v.label} for v in filters]
    # Assert
    h.assert_that(filter_results, h.has_items(
        h.has_entries({'count': 583, 'label': '[-0.11 - -0.05)'}),
        h.has_entries({'count': 682, 'label': '[-0.05 - -5.51E-3)'}),
        h.has_entries({'count': 605, 'label': '[-5.51E-3 - 0.02)'}),
        h.has_entries({'count': 726, 'label': '[0.02 - 0.05)'}),
        h.has_entries({'count': 660, 'label': '[0.05 - 0.11]'})
    ))


def test_column_partition_numerical_with_dominant():
    # Arrange
    df = pd.DataFrame(data={'col': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 8, 9, 10, 11]})
    dataset = Dataset(df, cat_features=[])
    # Act
    filters = partition_column(dataset, 'col', 5)
    filter_results = [{'count': v.filter(dataset.data).size, 'label': v.label} for v in filters]
    # Assert
    h.assert_that(filter_results, h.has_items(
        h.has_entries({'count': 16, 'label': '[1 - 5)'}),
        h.has_entries({'count': 3, 'label': '[5 - 8.9)'}),
        h.has_entries({'count': 3, 'label': '[8.9 - 11]'}),
    ))


def test_column_partition_categorical():
    # Arrange
    df = pd.DataFrame(data={'col': [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 4, 1, 1, 2, 3, 3, 8, 9, 10, 11]})
    dataset = Dataset(df, cat_features=['col'])
    # Act
    filters = partition_column(dataset, 'col', 3)
    filter_results = [{'count': v.filter(dataset.data).size, 'label': v.label} for v in filters]
    # Assert
    h.assert_that(filter_results, h.has_items(
        h.has_entries({'count': 9, 'label': '1'}),
        h.has_entries({'count': 5, 'label': '2'}),
        h.has_entries({'count': 3, 'label': '3'}),
        h.has_entries({'count': 5, 'label': 'Others'}),
    ))
