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
"""Contains unit tests for the missing_values_label_correlation check."""
import math

import numpy as np
import pandas as pd
from pytest import fixture

from deepchecks.tabular.checks.data_integrity.missing_values_label_correlation import MissingValuesLabelCorrelation
from deepchecks.tabular.dataset import Dataset


@fixture
def example_data():
    np.random.seed(30)
    x1 = np.random.random(size=400)
    x2 = np.random.choice(["blue", "red", "green"], size=400)
    labels = np.random.choice([0, 1], size=400)
    df = pd.DataFrame(zip(x1, x2, labels), columns=['x1', 'x2', 'label'])
    return df


def test_no_correlation_if_no_missing_values(example_data):
    """ if all values are present, there should be no correlation of missing values to the target"""

    result = MissingValuesLabelCorrelation().run(dataset=Dataset(example_data, label='label',
                                                                 cat_features=["x2"]))
    assert math.isclose(result.value["x1"], 0, abs_tol=1e-6)
    assert math.isclose(result.value["x2"], 0, abs_tol=1e-6)


def test_full_correlation(example_data):
    # flip all nan to 1 and all others to 0
    example_data.loc[:, "label"] = 0

    idx = np.random.choice([i for i in range(len(example_data))], size=int(len(example_data) / 10))
    example_data.loc[idx, "x1"] = np.NaN
    example_data.loc[idx, "label"] = 1

    result = MissingValuesLabelCorrelation().run(dataset=Dataset(example_data, label='label',
                                                                 cat_features=["x2"]))

    assert math.isclose(result.value["x1"], 1, abs_tol=1e-14), result.value["x1"]


def test_empty_string_is_nan(example_data):
    # flip all nan to 1 and all others to 0

    example_data.loc[:, "label"] = 0

    idx = np.random.choice([i for i in range(len(example_data))], size=int(len(example_data) / 10))
    example_data.loc[idx, "x2"] = ""
    example_data.loc[idx, "label"] = 1

    result = MissingValuesLabelCorrelation(empty_string_is_na=False).run(dataset=Dataset(example_data, label='label',
                                                                                         cat_features=["x2"]))

    assert math.isclose(result.value["x2"], 0, abs_tol=1e-14), result.value["x2"]

    result = MissingValuesLabelCorrelation(empty_string_is_na=True).run(dataset=Dataset(example_data, label='label',
                                                                                        cat_features=["x2"]))

    assert math.isclose(result.value["x2"], 1, abs_tol=1e-14), result.value["x2"]


def test_missing_values_are_correlated_to_label(example_data):

    # flip a few entries to nan and give those always the same label
    # so that missing and label are correlated
    idx = np.random.choice([i for i in range(len(example_data))], size=int(len(example_data) / 10))
    example_data.loc[idx, "x1"] = np.NaN
    example_data.loc[idx, "label"] = 1

    example_data.loc[idx, "x2"] = np.NaN

    result = MissingValuesLabelCorrelation().run(dataset=Dataset(example_data, label='label',
                                                                 cat_features=["x2"]))

    assert math.isclose(result.value["x1"], 0.09, abs_tol=1e-1), result.value["x1"]
    assert math.isclose(result.value["x2"], 0.09, abs_tol=1e-1), result.value["x2"]
