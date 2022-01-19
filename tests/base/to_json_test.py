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
"""to json tests"""
import jsonpickle
from hamcrest import assert_that, calling, raises, equal_to

from deepchecks.suites import full_suite
from deepchecks.checks import ColumnsInfo


def test_check_full_suite_not_failing(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    json_list = suite_res.get_result_json_list()
    assert isinstance(json_list, list)

def test_check_metadata(iris_dataset):
    check_res = ColumnsInfo(n_top_columns = 4).run(iris_dataset)
    json_res = jsonpickle.loads(check_res.to_json())
    assert_that(json_res['value'], 'Columns Info')
    assert_that(json_res['params'], 'n_top_columns = 4')
