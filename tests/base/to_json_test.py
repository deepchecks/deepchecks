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
"""to json tests"""
import jsonpickle
from hamcrest import assert_that, equal_to

from deepchecks.core.suite import SuiteResult
from deepchecks.tabular.checks import ColumnsInfo
from deepchecks.tabular.suites import full_suite


def _test_suite_json(json_res):
    json_suite_res = jsonpickle.loads(json_res)
    assert_that(json_suite_res['name'], equal_to('Full Suite'))
    assert_that(isinstance(json_suite_res['results'], list))
    for json_check_res in json_suite_res['results']:
        assert_that(isinstance(json_check_res, dict))

def test_check_full_suite_not_failing(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    json_res = suite_res.to_json()
    _test_suite_json(json_res)
    suite_from_json = SuiteResult.from_json(json_res)
    assert_that(isinstance(suite_from_json, SuiteResult))
    # reasserting the json
    _test_suite_json(suite_from_json.to_json())


def test_check_metadata(iris_dataset):
    check_res = ColumnsInfo(n_top_columns=4).run(iris_dataset)
    json_res = jsonpickle.loads(check_res.to_json())
    assert_that(json_res['value'], equal_to({
        'target': 'label',
        'sepal length (cm)': 'numerical feature',
        'sepal width (cm)': 'numerical feature',
        'petal length (cm)': 'numerical feature',
        'petal width (cm)': 'numerical feature'
    }))
    assert_that(json_res['check']['name'], equal_to('Columns Info'))
    assert_that(json_res['check']['params'], equal_to({'n_top_columns': 4}))
