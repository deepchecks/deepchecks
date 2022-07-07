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
"""Module with display utility functions."""
import base64

from deepchecks.utils.logger import get_logger

__all__ = ['imagetag', 'display_in_gui']

import sys
from io import StringIO

import pkg_resources


def imagetag(img: bytes) -> str:
    """Return html image tag with embedded image."""
    png = base64.b64encode(img).decode('ascii')
    return f'<img src="data:image/png;base64,{png}"/>'


def display_in_gui(result):
    """Display suite result or check result in a new python gui window."""
    try:
        required = {'pyqt5', 'pyqtwebengine'}
        # List of all packages installed (key is always in all small case!)
        installed = {pkg.key for pkg in list(pkg_resources.working_set)}
        missing = required - installed
        if missing:
            get_logger().warning('Missing packages in order to display result in GUI. either run "pip install %s"'
                                 ' or use "result.save_as_html()" to save result', {' '.join(missing)})
            return
        from PyQt5.QtWebEngineWidgets import QWebEngineView  # pylint: disable=import-outside-toplevel
        from PyQt5.QtWidgets import QApplication  # pylint: disable=import-outside-toplevel

        app = QApplication(sys.argv)

        web = QWebEngineView()
        web.setWindowTitle('deepchecks')
        web.setGeometry(0, 0, 1200, 1200)

        html_out = StringIO()
        result.save_as_html(html_out)
        web.setHtml(html_out.getvalue())
        web.show()

        sys.exit(app.exec_())
    except Exception:  # pylint: disable=broad-except
        get_logger().error('Unable to show result, run in an interactive environment'
                           ' or use "result.save_as_html()" to save result')
