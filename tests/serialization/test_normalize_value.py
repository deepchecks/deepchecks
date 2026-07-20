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
"""normalize_value serialization tests."""
import json

import pandas as pd
from hamcrest import assert_that, equal_to, instance_of

from deepchecks.core.serialization.common import normalize_value


def test_normalize_value_keeps_nested_dataframe_structure():
    """Nested CheckResult values (e.g. WeakSegmentsPerformance) must stay dict-shaped."""
    weak_segments = pd.DataFrame({
        'Score': [0.1, 0.2],
        'Feature1': ['a', 'b'],
    })
    value = {
        'weak_segments_list': weak_segments,
        'avg_score': 0.42,
    }

    normalized = normalize_value(value)

    assert_that(normalized, instance_of(dict))
    assert_that(normalized['avg_score'], equal_to(0.42))
    assert_that(normalized['weak_segments_list'], equal_to([
        {'Score': 0.1, 'Feature1': 'a'},
        {'Score': 0.2, 'Feature1': 'b'},
    ]))
    # Must be JSON-serializable without flattening nested structure into a string blob
    dumped = json.dumps(normalized)
    loaded = json.loads(dumped)
    assert_that(loaded['weak_segments_list'][0]['Feature1'], equal_to('a'))


def test_normalize_value_dataframe_returns_records_list():
    df = pd.DataFrame({'x': [1], 'y': [2]})
    assert_that(normalize_value(df), equal_to([{'x': 1, 'y': 2}]))
