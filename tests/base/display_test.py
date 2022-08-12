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
# pylint: disable=protected-access
import contextlib
import io
import pathlib
import typing as t
from unittest.mock import Mock, patch

import jsonpickle
import pandas as pd
import plotly.express
import plotly.io as pio
from bs4 import BeautifulSoup
from hamcrest import *
from ipywidgets import VBox, Widget

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks import ColumnsInfo, MixedNulls
from deepchecks.utils.json_utils import from_json
from tests.common import DummyCheck, create_check_result, create_suite_result, instance_of_ipython_formatter
from tests.serialization.test_check_result_serialization import (assert_check_result_html_output_structure,
                                                                 assert_plotly_loader, assert_style_loader)
from tests.serialization.test_suite_result_serialization import assert_suite_result_output_structure


def test_check_result_display():
    # Arrange
    with patch('deepchecks.core.display.display_html') as mock:
        result = create_check_result()
        result.show()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_check_result_html_output_structure(html)
        assert_plotly_loader(html, 'presence')
        assert_style_loader(html, 'presence')


def test_check_result_serialization_to_widget():
    # Arrange
    result = create_check_result()
    widget = result.to_widget()
    # Assert
    assert_that(widget, instance_of(VBox))
    assert_that(widget.children, has_length(4))


def test_check_result_serialization_to_json(iris_dataset):
    import json

    result = ColumnsInfo(n_top_columns=4).run(iris_dataset)
    serialized_result = result.to_json()

    assert_that(serialized_result, instance_of(str))
    json.loads(serialized_result)  # must not faile


def test_check_result_repr(iris_dataset):
    # Arrange
    check = MixedNulls()
    check_res = check.run(iris_dataset)

    # Assert
    assert_that(check.__repr__(), is_('MixedNulls'))
    assert_that(
        check_res.__repr__(),
        equal_to(
            'Mixed Nulls: {\'sepal length (cm)\': {}, \'sepal width (cm)\': {}, '
            '\'petal length (cm)\': {}, \'petal width (cm)\': {}, \'target\': {}}'
        )
    )


def test_check_result_init():
    assert_that(
        calling(CheckResult).with_args(value=None, display={}),
        raises(DeepchecksValueError, 'Can\'t display item of type: <class \'dict\'>')
    )


def test_check_result_deserialization_from_json(iris):
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


def test_check_result_show_with_sphinx_gallery_env_enabled():
    with plotly_default_renderer('sphinx_gallery'):
        # Arrange
        check_result = create_check_result(10)
        # Assert
        r = check_result.show()
        assert_that(hasattr(r, '_repr_html_'))
        html = r._repr_html_()
        assert_that(html, instance_of(str))
        assert_check_result_html_output_structure(html)
        assert_plotly_loader(html, 'presence')
        assert_style_loader(html, 'presence')


def test_check_result_display_with_enabled_colab_enviroment():
    # Arrange
    result = create_check_result(value=[10, 20, 30])
    # Assert
    with patch('deepchecks.core.display.is_colab_env', return_value=True):
        with patch('deepchecks.core.display.display_html') as mock:
            result.show()
            mock.assert_called_once()

            args, kwargs = list(mock.call_args)
            html, *_ = args
            assert_that(html, instance_of(str))

            soup = BeautifulSoup(html, 'html.parser')
            html_tag = soup.select_one('html')
            body_tag = soup.select_one('body')
            head_tag = soup.select_one('head')
            style_tag = soup.select_one('head > style')
            script_tag = soup.select_one('head > script')

            assert_that(html_tag, not_none())
            assert_that(body_tag, not_none())
            assert_that(head_tag, not_none())
            assert_that(style_tag, not_none())
            assert_that(script_tag, not_none())
            assert_check_result_html_output_structure(soup)


def test_check_result_ipython_display():
    # Arrange
    result = create_check_result(value=[10, 20, 30])
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        result._ipython_display_()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_that(html, instance_of(str))
        assert_check_result_html_output_structure(html)
        assert_plotly_loader(html, 'presence')
        assert_style_loader(html, 'presence')


def test_check_result_ipython_display():
    # Arrange
    result = create_check_result(value=[10, 20, 30])
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        result._ipython_display_()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_that(html, instance_of(str))
        assert_check_result_html_output_structure(html)
        assert_plotly_loader(html, 'presence')
        assert_style_loader(html, 'presence')


def test_check_result_display_in_iframe():
    # Arrange
    result = create_check_result(value=[10, 20, 30])
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        result.show_in_iframe()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args

        assert_that(html, instance_of(str))
        soup = BeautifulSoup(html, 'html.parser')
        iframe = soup.select_one('iframe')

        assert_that(iframe, not_none())
        assert_that(iframe.get('srcdoc'), not_none())


def test_check_result_not_interactive_display():
    # Arrange
    result = create_check_result(value=[10, 20, 30])
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        result.show_not_interactive()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args

        assert_that(html, instance_of(str))
        soup = BeautifulSoup(html, 'html.parser')
        assert_check_result_html_output_structure(soup)

        scripts = soup.select('scripts')
        assert_that(scripts, has_length(equal_to(0)))


def test_check_result_repr_mimebundle():
    # Arrange
    result = create_check_result(value=10)
    # Assert
    assert_that(
        result._repr_mimebundle_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)),
            has_entries({
                'text/html': instance_of(str),
                'application/json': any_of(instance_of(dict), instance_of(list))
            }))
    )


def test_check_result_repr_html():
    # Arrange
    result = create_check_result(value=10)
    # Assert
    html = result._repr_html_()
    assert_that(html, instance_of(str))
    assert_check_result_html_output_structure(html)


def test_check_result_repr_json():
    # Arrange
    result = create_check_result(value={'foo': 10, 'bar': 20})
    # Assert
    assert_that(
        result._repr_json_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)))
    )


def test_check_result_save_as_html():
    # Arrange
    result = create_check_result(value=10)
    # Act
    filename = t.cast(str, result.save_as_html('check-result.html'))
    # Assert
    assert_saved_html_file(filename)


def test_check_result_save_as_html_without_providing_output_filename():
    # Arrange
    result = create_check_result(value=10)
    # Act
    filename = t.cast(str, result.save_as_html())
    # Assert
    assert_saved_html_file(filename)


def test_check_result_save_as_html_with_as_widget_parameter_set_to_false():
    # Arrange
    result = create_check_result(value=10)
    # Act
    filename = t.cast(str, result.save_as_html(as_widget=False))
    # Assert
    assert_saved_html_file(filename)


def test_check_result_save_as_html_with_iobuffer_passed_to_file_parameter():
    # Arrange
    result = create_check_result(value=10)
    buffer = io.StringIO('')
    # Act
    result.save_as_html(buffer)
    # Assert
    buffer.seek(0)
    assert_that(buffer.read(), is_html_document())
    buffer.close()


# ==========================================================


def test_check_failure_to_widget():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    assert_that(failure.to_widget(), instance_of(Widget))


def test_check_failure_display_with_enabled_colab_enviroment():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    with patch('deepchecks.core.display.is_colab_env', return_value=True):
        with patch('deepchecks.core.display.display_html') as mock:
            failure.show()
            mock.assert_called_once()
            args, kwargs = list(mock.call_args)
            html, *_ = args
            assert_that(html,instance_of(str))


def test_check_failure_display():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        failure.show()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_that(html, instance_of(str))


def test_check_failure_show_with_sphinx_gallery_env_enabled():
    with plotly_default_renderer('sphinx_gallery'):
        # Arrange
        failure = CheckFailure(DummyCheck(), Exception('error message'))
        # Assert
        r = failure.show()
        assert_that(hasattr(r, '_repr_html_'))
        assert_that(r._repr_html_(), instance_of(str))


def test_check_failure_ipython_display():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        failure._ipython_display_()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_that(html, instance_of(str))


def test_check_failure_repr_mimebundle():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    assert_that(
        failure._repr_mimebundle_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)),
            has_entries({
                'text/html': instance_of(str),
                'application/json': any_of(instance_of(dict), instance_of(dict))}))
    )


def test_check_failure_repr_html():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    assert_that(failure._repr_html_(), all_of(instance_of(str)))


def test_check_failure_repr_json():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    assert_that(
        failure._repr_json_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)))
    )


def test_check_failure_save_as_html():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Act
    filename = t.cast(str, failure.save_as_html('check-failure.html'))
    # Assert
    assert_saved_html_file(filename)


def test_check_failure_save_as_html_without_providing_output_filename():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Act
    filename = t.cast(str, failure.save_as_html())
    # Assert
    assert_saved_html_file(filename)


def test_check_failure_save_as_html_with_as_widget_parameter_set_to_false():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Act
    filename = t.cast(str, failure.save_as_html(as_widget=False))
    # Assert
    assert_saved_html_file(filename)


def test_check_failure_save_as_html_with_iobuffer_passed_to_file_parameter():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    buffer = io.StringIO('')
    # Act
    failure.save_as_html(buffer)
    # Assert
    buffer.seek(0)
    assert_that(buffer.read(), is_html_document())
    buffer.close()


# # ==========================================================


def test_suite_result_to_widget():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    assert_that(suite_result.to_widget(), instance_of(Widget))


def test_suite_result_show():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        suite_result.show()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_suite_result_output_structure(html)
        assert_style_loader(html, 'presence')
        assert_plotly_loader(html, 'presence')


def test_suite_result_ipython_display():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        suite_result._ipython_display_()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args
        assert_suite_result_output_structure(html)
        assert_style_loader(html, 'presence')
        assert_plotly_loader(html, 'presence')


def test_suite_result_display_in_iframe():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        suite_result.show_in_iframe()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args

        assert_that(html, instance_of(str))
        soup = BeautifulSoup(html, 'html.parser')
        iframe = soup.select_one('iframe')

        assert_that(iframe, not_none())
        assert_that(iframe.get('srcdoc'), not_none())


def test_suite_result_not_interactive_display():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    with patch('deepchecks.core.display.display_html') as mock:
        suite_result.show_not_interactive()
        mock.assert_called_once()
        args, kwargs = list(mock.call_args)
        html, *_ = args

        assert_that(html, instance_of(str))
        soup = BeautifulSoup(html, 'html.parser')
        assert_suite_result_output_structure(soup)

        scripts = soup.select('scripts')
        assert_that(scripts, has_length(equal_to(0)))


def test_suite_result_ipython_display_with_colab_env_enabled():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    with patch('deepchecks.core.display.is_colab_env', Mock(return_value=True)):
        with patch('deepchecks.core.display.display_html') as mock:
            suite_result._ipython_display_()
            mock.assert_called_once()
            args, kwargs = list(mock.call_args)
            html, *_ = args
            assert_that(html, instance_of(str))

            soup = BeautifulSoup(html, 'html.parser')
            html_tag = soup.select_one('html')
            body_tag = soup.select_one('body')
            head_tag = soup.select_one('head')
            style_tag = soup.select_one('head > style')
            script_tag = soup.select_one('head > script')
            assert_that(html_tag, not_none())
            assert_that(body_tag, not_none())
            assert_that(head_tag, not_none())
            assert_that(style_tag, not_none())
            assert_that(script_tag, not_none())


def test_suite_result_repr_mimebundle():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    assert_that(
        suite_result._repr_mimebundle_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)),
            has_entries({
                'text/html': instance_of(str),
                'application/json': any_of(instance_of(dict), instance_of(list))
            }))
    )


def test_suite_result_repr_html():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    assert_that(
        suite_result._repr_html_(),
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            is_html_document()
        )
    )


def test_suite_result_repr_json():
    # Arrange
    suite_result = create_suite_result()
    # Assert
    assert_that(
        suite_result._repr_json_(),
        all_of(
            instance_of(dict),
            has_length(greater_than(0)))
    )


def test_suite_resul_save_as_html():
    # Arrange
    suite_result = create_suite_result()
    # Act
    filename = t.cast(str, suite_result.save_as_html('suite_result.html'))
    # Assert
    assert_saved_html_file(filename)


def test_suite_result_save_as_html_without_providing_output_filename():
    # Arrange
    suite_result = create_suite_result()
    # Act
    filename = t.cast(str, suite_result.save_as_html())
    # Assert
    assert_saved_html_file(filename)


def test_suite_result_save_as_html_with_as_widget_parameter_set_to_false():
    # Arrange
    suite_result = create_suite_result()
    # Act
    filename = t.cast(str, suite_result.save_as_html(as_widget=False))
    # Assert
    assert_saved_html_file(filename)


def test_suite_result_save_as_html_with_iobuffer_passed_to_file_parameter():
    # Arrange
    suite_result = create_suite_result()
    buffer = io.StringIO('')
    # Act
    suite_result.save_as_html(buffer)
    # Assert
    buffer.seek(0)
    assert_that(buffer.read(), is_html_document())
    buffer.close()


# ==========================================================


def assert_saved_html_file(filename='output.html'):
    output = pathlib.Path(filename)
    assert_that(output.exists() and output.is_file())
    try:
        content = output.open('r', encoding='utf-8').read()
        soup = BeautifulSoup(content, 'html.parser')
        html_tag = soup.select_one('html')
        head_tag = soup.select_one('html > head')
        body_tag = soup.select_one('html > body')
        assert_that(html_tag, not_none())
        assert_that(head_tag, not_none())
        assert_that(body_tag, not_none())
    finally:
        output.unlink()


def is_html_document():
    any_whitespace = r'[\s]*'
    anything = r'[\s\S\d\D\w\W]*'
    regexp = (
        fr'^{any_whitespace}({anything})<html( lang="en")*>{any_whitespace}'
        fr'<head>({anything})<\/head>{any_whitespace}'
        fr'<body({anything})>({anything})<\/body>{any_whitespace}'
        fr'<\/html>{any_whitespace}$'
    )
    return all_of(instance_of(str), matches_regexp(regexp))


@contextlib.contextmanager
def plotly_default_renderer(name):
    original_renderer = pio.renderers.default
    pio.renderers.default = name
    yield
    pio.renderers.default = original_renderer
