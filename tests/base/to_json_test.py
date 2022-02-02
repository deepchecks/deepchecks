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
"""to json tests"""
import jsonpickle
from hamcrest import assert_that, equal_to

from deepchecks.tabular.suites import full_suite
from deepchecks.tabular.checks import ColumnsInfo


def test_check_full_suite_not_failing(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    json_res = jsonpickle.loads(suite_res.to_json())
    assert_that(json_res['name'], equal_to('Full Suite'))
    assert isinstance(json_res['results'], list)


def test_check_metadata(iris_dataset):
    check_res = ColumnsInfo(n_top_columns=4).run(iris_dataset)
    json_res = jsonpickle.loads(check_res.to_json())
    assert_that(json_res['value'], equal_to({'sepal length (cm)': 'numerical feature',
                                            'sepal width (cm)': 'numerical feature',
                                            'petal length (cm)': 'numerical feature',
                                            'petal width (cm)': 'numerical feature',
                                            'target': 'label'}))
    assert_that(json_res['name'], equal_to('Columns Info'))
    assert_that(json_res['params'], equal_to({'n_top_columns': 4}))
