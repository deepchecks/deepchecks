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
"""Tests for Feature Feature Correlation check"""
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_items, has_length

from deepchecks.tabular.checks.data_integrity.feature_feature_correlation import FeatureFeatureCorrelation
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.datasets.classification import adult

adult_data = adult.load_data(as_train_test=False)


def test_feature_feature_correlation():
    result = FeatureFeatureCorrelation().run(adult_data)
