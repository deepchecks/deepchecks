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
"""IPython utilities tests."""
import io
from unittest.mock import patch

import pytest
import tqdm
from hamcrest import assert_that, contains_exactly, equal_to, has_entries, instance_of, none, not_none

from deepchecks.utils import ipython


def test_progress_bar_creation():
    with patch('deepchecks.utils.ipython.is_zmq_interactive_shell', return_value=True):
        with patch('deepchecks.utils.ipython.is_widgets_enabled', return_value=True):
            assert_that(
                ipython.create_progress_bar(
                    iterable=list(range(10)),
                    name='Dummy',
                    unit='D'
                ),
                instance_of(tqdm.notebook.tqdm)
            )


def test_progress_bar_creation_with_disabled_widgets():
    with patch('deepchecks.utils.ipython.is_zmq_interactive_shell', return_value=True):
        with patch('deepchecks.utils.ipython.is_widgets_enabled', return_value=False):
            assert_that(
                ipython.create_progress_bar(
                    iterable=list(range(10)),
                    name='Dummy',
                    unit='D'
                ),
                instance_of(ipython.PlainNotebookProgressBar)
            )


def test_progress_bar_creation_in_not_notebook_env():
    with patch('deepchecks.utils.ipython.is_zmq_interactive_shell', return_value=False):
        assert_that(
            ipython.create_progress_bar(
                iterable=list(range(10)),
                name='Dummy',
                unit='D'
            ),
            instance_of(tqdm.tqdm)
        )


def test_dummy_progress_bar_creation():
    dummy_progress_bar = ipython.DummyProgressBar(name='Dummy')
    assert_that(len(list(dummy_progress_bar.pb)) == 1)


def test_get_jupyter_server_url():
    server_url = 'http://localhost:8888/?token=qwert'
    server_root_directory = '/home/user/Projects/deepchecks/'
    cmd_output = f'Currently running servers:\n{server_url} :: {server_root_directory}'

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        url = ipython.get_jupyter_server_url()
        assert_that(url == server_url)
        mock.assert_called_once_with('jupyter server list')


def test_get_jupyter_server_url_with_output_that_does_not_contain_url():
    cmd_output = 'Currently running servers:'

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        assert_that(ipython.get_jupyter_server_url() is None)
        mock.assert_called_once_with('jupyter server list')


def test_get_jupyter_server_url_with_multiple_urls_in_output():
    cmd_output = (
        'Currently running servers:\n'
        'http://localhost:8888/?token=qwert :: /home/user/Projects/deepchecks/\n'
        'http://localhost:8889/?token=asdfg :: /home/user/Projects/deepchecks-v2/\n'
    )

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        assert_that(ipython.get_jupyter_server_url() is None)
        mock.assert_called_once_with('jupyter server list')


def test_get_jupyter_server_url_with_output_of_unknown_format():
    cmd_output = 'Server url is - http://localhost:8889/?token=asdfg'

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        assert_that(ipython.get_jupyter_server_url() is None)
        mock.assert_called_once_with('jupyter server list')


def test_extract_jupyter_server_token():
    token = 'qwert'
    extracted_token = ipython.extract_jupyter_server_token(f'token={token}&a=10&b=20')
    assert_that(extracted_token, equal_to(token))


def test_extract_jupyter_server_token_with_provided_empty_string():
    extracted_token = ipython.extract_jupyter_server_token('')
    assert_that(extracted_token, equal_to(''))


def test_get_jupyterlab_extensions__verifying_configs_merging():
    # NOTE:
    # user configs have the highest priority thefore (they appear first in the output)
    # after configs merging 'jupyterlab-plotly' must be with status - enabled
    cmd_output = (
        'JupyterLab v3.4.2\n'
        '/home/user/.local/share/jupyter/labextensions\n'
        '   jupyterlab-plotly v5.5.0 enabled OK\n'
        '\n'
        '/home/user/Projects/deepchecks/venv/share/jupyter/labextensions\n'
        '   jupyterlab-plotly v5.5.0 disabled OK\n'
    )

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        extensions = ipython.get_jupyterlab_extensions(merge=True)

        assertion = has_entries({
            'jupyterlab-plotly': has_entries({
                'name': 'jupyterlab-plotly',
                'enabled': True,
                'status': 'OK',
                'installed_version': 'v5.5.0'
            })
        })

        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, assertion)


def test_get_jupyterlab_extensions__verifying_configs_merging_v2():
    # opposite situation to the 'test_get_jupyterlab_extensions__verifying_configs_merging'
    cmd_output = (
        'JupyterLab v3.4.2\n'
        '/home/user/.local/share/jupyter/labextensions\n'
        '   jupyterlab-plotly v5.5.0 disabled OK\n'
        '\n'
        '/home/user/Projects/deepchecks/venv/share/jupyter/labextensions\n'
        '   jupyterlab-plotly v5.5.0 enabled OK\n'
    )

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        extensions = ipython.get_jupyterlab_extensions(merge=True)

        assertion = has_entries({
            'jupyterlab-plotly': has_entries({
                'name': 'jupyterlab-plotly',
                'enabled': False,
                'status': 'OK',
                'installed_version': 'v5.5.0'
            })
        })

        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, assertion)


def test_get_jupyterlab_extensions_with_empty_output_from_cmd():
    cmd_output = ''

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        extensions = ipython.get_jupyterlab_extensions()
        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, equal_to({}))


def test_get_jupyterlab_extensions_with_output_of_unknown_format():
    cmd_output = (
        'Extensions:\n'
        '   jupyterlab-plotly v5.5.0 disabled OK\n'
        '   jupyterlab-plotly v5.5.0 enabled OK\n'
    )
    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        extensions = ipython.get_jupyterlab_extensions()
        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, equal_to({}))


def test_get_jupyterlab_extensions_with_raising_exception():
    with patch('subprocess.getoutput', side_effect=RuntimeError('Exception!')) as mock:
        extensions = ipython.get_jupyterlab_extensions()
        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, none())


def test_get_jupyterlab_extensions_with_merge_parameter_set_to_false():
    cmd_output = (
        'JupyterLab v3.4.2\n'
        '/home/user/.local/share/jupyter/labextensions\n'
        '   jupyterlab-plotly v5.5.0 enabled OK\n'
        '   @jupyter-widgets/jupyterlab-manager v3.0.1 disabled OK (python, jupyterlab_widgets)\n'
        '\n'
        '/home/user/Projects/deepchecks/venv/share/jupyter/labextensions\n'
        '   jupyterlab_pygments v0.2.2 disabled OK (python, jupyterlab_pygments)\n'
        '   catboost-widget v1.0.0 enabled OK\n'
    )

    with patch('subprocess.getoutput', return_value=cmd_output) as mock:
        extensions = ipython.get_jupyterlab_extensions(merge=False)

        assertion = has_entries({
            '/home/user/.local/share/jupyter/labextensions': contains_exactly(
                has_entries({
                    'name': 'jupyterlab-plotly',
                    'enabled': True,
                    'status': 'OK',
                    'installed_version': 'v5.5.0'
                }),
                has_entries({
                    'name': '@jupyter-widgets/jupyterlab-manager',
                    'enabled': False,
                    'status': 'OK',
                    'installed_version': 'v3.0.1'
                }),
            ),
            '/home/user/Projects/deepchecks/venv/share/jupyter/labextensions': contains_exactly(
                has_entries({
                    'name': 'jupyterlab_pygments',
                    'enabled': False,
                    'status': 'OK',
                    'installed_version': 'v0.2.2'
                }),
                has_entries({
                    'name': 'catboost-widget',
                    'enabled': True,
                    'status': 'OK',
                    'installed_version': 'v1.0.0'
                }),
            ),
        })

        mock.assert_called_once_with('jupyter labextension list')
        assert_that(extensions, assertion)


def test_request_jupyterlab_extensions(jupyterlab_extensions):  # pylint: disable=redefined-outer-name
    extensions_string, assertion = jupyterlab_extensions
    return_value = io.StringIO(extensions_string)

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/lab/api/extensions?token=asdfg'
        extensions = ipython.request_jupyterlab_extensions(server_url)

        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, not_none())
        assert_that(list(extensions.values()), assertion)


def test_request_jupyterlab_extensions_with_empty_output():
    return_value = io.StringIO('[]')

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/lab/api/extensions?token=asdfg'
        extensions = ipython.request_jupyterlab_extensions(server_url)

        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, equal_to({}))


def test_request_jupyterlab_extensions_with_output_of_unknown_format():
    return_value = io.StringIO('{"hello": "world"}')

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/lab/api/extensions?token=asdfg'
        extensions = ipython.request_jupyterlab_extensions(server_url)

        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, none())


def test_request_nbclassic_extensions(nbclassic_extensions):  # pylint: disable=redefined-outer-name
    extensions_string, assertion = nbclassic_extensions
    return_value = io.StringIO(extensions_string)

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/api/config/notebook?token=asdfg'
        extensions = ipython.request_nbclassic_extensions(server_url)
        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, assertion)


def test_request_nbclassic_extensions_with_empty_output():
    return_value = io.StringIO('{}')

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/api/config/notebook?token=asdfg'
        extensions = ipython.request_nbclassic_extensions(server_url)
        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, equal_to({}))


def test_request_nbclassic_extensions_with_output_of_unknown_format():
    return_value = io.StringIO('{"hello": "world"}')

    with patch('urllib.request.urlopen', return_value=return_value) as mock:
        server_url = 'http://localhost:8889/?token=asdfg'
        extensions_url = 'http://localhost:8889/api/config/notebook?token=asdfg'
        extensions = ipython.request_nbclassic_extensions(server_url)
        mock.assert_called_once_with(extensions_url)
        assert_that(extensions, none())


# =========================================================

@pytest.fixture
def jupyterlab_extensions():
    string = """
    [
        {
            "name": "jupyterlab-plotly",
            "enabled": false,
            "installed_version": "5.5.0",
            "status": "ok"
        },
        {
            "name": "@jupyter-widgets/jupyterlab-manager",
            "enabled": true,
            "installed_version": "3.0.1",
            "status": "ok"
        }
    ]
    """
    assertion = contains_exactly(
        has_entries({
            'name': 'jupyterlab-plotly',
            'enabled': False,
            'status': 'ok',
            'installed_version': '5.5.0'
        }),
        has_entries({
            'name': '@jupyter-widgets/jupyterlab-manager',
            'enabled': True,
            'status': 'ok',
            'installed_version': '3.0.1'
        }),
    )
    return string, assertion


@pytest.fixture
def nbclassic_extensions():
    string = """
    {
        "load_extensions": {
            "catboost-widget/extension": true,
            "jupyterlab-plotly/extension": false,
            "jupyter-js-widgets/extension": true
        }
    }
    """
    assertion = has_entries({
        'catboost-widget': has_entries({
            'name': 'catboost-widget',
            'enabled': True
        }),
        'jupyterlab-plotly': has_entries({
            'name': 'jupyterlab-plotly',
            'enabled': False
        }),
        'jupyter-js-widgets': has_entries({
            'name': 'jupyter-js-widgets',
            'enabled': True
        }),
    })
    return string, assertion
