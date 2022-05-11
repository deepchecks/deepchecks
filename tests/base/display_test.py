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
"""display tests"""
from unittest.mock import Mock, patch

import jsonpickle
import pandas as pd
import plotly.express
import plotly.io as pio
from hamcrest import (assert_that, calling, has_length, instance_of, is_,
                      not_none, raises)
from ipywidgets import VBox
from plotly.graph_objs import FigureWidget

from deepchecks.core.check_result import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks import ColumnsInfo, DataDuplicates, MixedNulls
from deepchecks.utils.json_utils import from_json

pio.renderers.default = "json"


# display check
def test_check_run_display(iris_dataset):
    # Arrange
    with patch('deepchecks.core.check_result.display') as mock:
        check_res = ColumnsInfo(n_top_columns=4).run(iris_dataset)
        assert_that(check_res.display_check(), is_(None))
        mock.assert_called_once()


def test_check_run_display_as_widget(iris_dataset):
    with patch('deepchecks.core.check_result.is_widgets_enabled', Mock(return_value=True)):
        # Arrange
        check_res = ColumnsInfo(n_top_columns=4).run(iris_dataset)
        dispaly_box = check_res.display_check(as_widget=True)
        # Assert
        assert_that(dispaly_box, instance_of(VBox))
        assert_that(dispaly_box.children, has_length(4))


def test_check_run_display_unique_id(iris_dataset):
    with patch('deepchecks.core.check_result.display') as mock:
        # Arrange
        check_res = ColumnsInfo(n_top_columns=4).run(iris_dataset)
        # Assert
        assert_that(check_res.display_check(unique_id='qwerty'), is_(None))
        mock.assert_called_once()


def test_check_run_display_condition(iris_dataset):
    with patch('deepchecks.core.check_result.display') as mock:
        # Arrange
        check_res = DataDuplicates().add_condition_ratio_not_greater_than(0).run(iris_dataset)
        # Assert
        assert_that(check_res.display_check(unique_id='qwerty'), is_(None))
        mock.assert_called_once()


def test_check_run_display_nothing_to_show(iris_dataset):
    with patch('deepchecks.core.check_result.display') as mock:
        # Arrange
        check_res = MixedNulls().run(iris_dataset)
        # Assert
        check_res.display_check(unique_id='qwerty')
        mock.assert_called_once()


def test_check_result_repr(iris_dataset):
    # Arrange
    check = MixedNulls()
    check_res = check.run(iris_dataset)

    # Assert
    assert_that(check.__repr__(), is_('MixedNulls'))
    assert_that(check_res.__repr__(), is_('Mixed Nulls: defaultdict(<class \'dict\'>, {})'))


def test_check_result_init(iris_dataset):
    assert_that(calling(CheckResult).with_args(value=None, display={}),
                raises(DeepchecksValueError, 'Can\'t display item of type: <class \'dict\'>'))


def test_check_result_display_plt_func():
    # Arrange
    def display_func():
        return 'test'
    check_res = CheckResult(value=7, header='test', display=[display_func])
    check_res.check = DataDuplicates()

    # Assert
    with patch('deepchecks.core.check_result.display') as mock:
        assert_that(check_res.display_check(), is_(None))
        mock.assert_called_once()
    with patch('deepchecks.core.check_result.is_widgets_enabled', Mock(return_value=True)):
        assert_that(check_res.display_check(as_widget=True), not_none())


def test_check_result_display_plotly(iris):
    # Arrange
    plot = plotly.express.bar(iris)
    check_res = CheckResult(value=7, header='test', display=[plot])
    check_res.check = DataDuplicates()

    with patch('deepchecks.core.check_result.is_widgets_enabled', Mock(return_value=True)):
        display = check_res.display_check(as_widget=True)

    # Assert
    assert_that(display, instance_of(VBox))
    assert_that(display.children[3], instance_of(VBox)) # additional output wiidget
    assert_that(display.children[3].children[1], instance_of(FigureWidget)) # plotly figure widget


def test_check_result_to_json():
    # Arrange
    check_res = CheckResult(value=7, header='test', display=['hi'])
    check_res.display = [{}]
    check_res.check = DataDuplicates()

    # Assert
    assert_that(calling(check_res.to_json).with_args(),
                raises(Exception, "Unable to handle display item of type: <class 'dict'>"))


def test_check_result_from_json(iris):
    # Arrange
    plot = plotly.express.bar(iris)

    json_to_display = jsonpickle.dumps({
        'type': 'CheckResult',
        'check': {
            'type': 'DummyCheckClass',
            'name': 'Dummy Check',
            'summary': 'summary',
            'params': {}
        },
        'display': [
            {'type': 'html', 'payload': 'hi'},
            {'type': 'plotly', 'payload': plot.to_json()},
            {
                'type': 'dataframe', 
                'payload': pd.DataFrame({'a': [1, 2], 'b': [1, 2]}).to_dict(orient='records')
            },
        ],
        'header': 'header',
        'value': 5,
    })

    # Assert
    assert_that(isinstance(from_json(json_to_display), CheckResult))


def test_check_result_show():
    # Arrange
    cr = CheckResult(value=0, header='test', display=[''])
    cr.check = DataDuplicates()
    
    with patch('deepchecks.core.check_result.is_notebook', Mock(return_value=True)):
        with patch('deepchecks.core.check_result.is_widgets_enabled', Mock(return_value=True)):
            with patch('deepchecks.core.check_result.display_html') as mock:
                # Assert
                assert_that(cr.show(), is_(None))
                mock.assert_called_once()
