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
# pylint: disable=unused-argument
"""Module with check/suite result display strategy in different envs."""
import abc
import io
import pathlib
import sys
import typing as t
from multiprocessing import Process

import plotly.io as pio
from IPython.core.display import display_html
from ipywidgets import Widget

from deepchecks.core.serialization.abc import HTMLFormatter, HtmlSerializer, WidgetSerializer
from deepchecks.utils.ipython import is_colab_env
from deepchecks.utils.logger import get_logger
from deepchecks.utils.strings import create_new_file_name, get_random_string, widget_to_html

if t.TYPE_CHECKING:
    from wandb.sdk.data_types.base_types.wb_value import WBValue  # pylint: disable=unused-import

__all__ = ['DisplayableResult', 'save_as_html', 'display_in_gui']


T = t.TypeVar('T')


class DisplayableResult(abc.ABC):
    """Display API for the check/suite result objects."""

    @property
    @abc.abstractmethod
    def widget_serializer(self) -> WidgetSerializer[t.Any]:
        """Return WidgetSerializer instance."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def html_serializer(self) -> HtmlSerializer[t.Any]:
        """Return HtmlSerializer instance."""
        raise NotImplementedError()

    def show(
        self,
        unique_id: t.Optional[str] = None,
        **kwargs
    ) -> t.Optional[HTMLFormatter]:
        """Display result.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwargs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method

        Returns
        -------
        Optional[HTMLFormatter] :
            when used by sphinx-gallery
        """
        output_id = unique_id or get_random_string(n=25)

        if 'sphinx_gallery' in pio.renderers.default:
            # TODO: why we need this? add comments
            html = self.html_serializer.serialize(output_id=output_id, **kwargs)

            class TempSphinx:
                def _repr_html_(self):
                    return html

            return TempSphinx()

        if is_colab_env():
            display_html(
                self.html_serializer.serialize(
                    full_html=True,
                    is_for_iframe_with_srcdoc=True,
                    collapsible=True,
                    **kwargs
                ),
                raw=True
            )
        else:
            display_html(
                self.html_serializer.serialize(
                    output_id=output_id,
                    **kwargs
                ),
                raw=True
            )

    def show_in_iframe(
        self,
        unique_id: t.Optional[str] = None,
        connected: bool = False,
        **kwargs
    ):
        """Display result in an iframe.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwargs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method
        """
        output_id = unique_id or get_random_string(n=25)

        if is_colab_env():
            display_html(
                self.html_serializer.serialize(
                    full_html=True,
                    collapsible=True,
                    **kwargs
                ),
                raw=True
            )
        else:
            display_html(
                self.html_serializer.serialize(
                    output_id=output_id,
                    embed_into_iframe=True,
                ),
                raw=True
            )

    def show_in_window(self, **kwargs):
        """Display result in a separate window."""
        display_in_gui(self)

    def show_not_interactive(
        self,
        unique_id: t.Optional[str] = None,
        **kwargs
    ):
        """Display the not interactive version of result output.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwargs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method
        """
        output_id = unique_id or get_random_string(n=25)

        if is_colab_env():
            display_html(
                self.html_serializer.serialize(
                    full_html=True,
                    use_javascript=False,
                    **kwargs
                ),
                raw=True
            )
        else:
            display_html(
                self.html_serializer.serialize(
                    output_id=output_id,
                    use_javascript=False,
                    **kwargs
                ),
                raw=True
            )

    def _ipython_display_(self, **kwargs):
        """Display result.."""
        self.show(**kwargs)

    @abc.abstractmethod
    def to_widget(self, **kwargs) -> Widget:
        """Serialize result into a ipywidgets.Widget instance."""
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self, **kwargs) -> str:
        """Serialize result into a json string."""
        raise NotImplementedError()

    @abc.abstractmethod
    def to_wandb(self, **kwargs) -> 'WBValue':
        """Send result to the wandb."""
        raise NotImplementedError()

    @abc.abstractmethod
    def save_as_html(
        self,
        file: t.Union[str, io.TextIOWrapper, None] = None,
        **kwargs
    ) -> t.Optional[str]:
        """Save a result to an HTML file.

        Parameters
        ----------
        file : filename or file-like object
            The file to write the HTML output to. If None writes to output.html

        Returns
        -------
        Optional[str] :
            name of newly create file
        """
        raise NotImplementedError()


def display_in_gui(result: DisplayableResult):
    """Display suite result or check result in a new pyqt5 gui window.

    NOTE: regarding 'QWebEngineView().setHtml()'
    Content larger than 2 MB cannot be displayed, because converts the
    provided HTML to percent-encoding and places data : in front of it
    to create the URL that it navigates to. Thereby, the provided code
    becomes a URL that exceeds the 2 MB limit set by Chromium. If the
    content is too large, the loadFinished() signal is triggered with
    success=false.
    Link: "https://doc.qt.io/qtforpython-5/PySide2/QtWebEngineWidgets
    /QWebEngineView.html#PySide2.QtWebEngineWidgets.PySide2.QtWebEngineWidgets
    .QWebEngineView.setHtml"

    Plotly src code is near 3 MB, as result Suite/Check result output
    cannot be displayed with inlined plotlyjs library.
    """
    try:
        from PyQt5.QtCore import QUrl  # pylint: disable=import-outside-toplevel
        from PyQt5.QtWebEngineWidgets import QWebEngineView  # pylint: disable=import-outside-toplevel
        from PyQt5.QtWidgets import QApplication  # pylint: disable=import-outside-toplevel
    except ImportError:
        get_logger().error(
            'Missing packages in order to display result in GUI, '
            'either run "pip install pyqt5, pyqtwebengine" '
            'or use "result.save_as_html()" to save result'
        )
    else:
        filename = t.cast(str, result.save_as_html('deepchecks-report.html'))
        filepath = pathlib.Path(filename).absolute()

        def app(filename: str):
            filepath = pathlib.Path(filename)
            try:
                app = QApplication.instance()
                if app is None:
                    app = QApplication([])
                    app.lastWindowClosed.connect(app.quit)
                web = QWebEngineView()
                web.setWindowTitle('deepchecks')
                web.setGeometry(0, 0, 1200, 1200)
                web.load(QUrl.fromLocalFile(str(filepath)))
                web.show()
                sys.exit(app.exec_())
            finally:
                filepath.unlink()

        Process(target=app, args=(str(filepath),)).start()


def get_result_name(result) -> str:
    """Get Check/Suite result instance name."""
    if hasattr(result, 'name'):
        return result.name
    elif hasattr(result, 'get_header') and callable(getattr(result, 'get_header')):
        return result.get_header()
    else:
        return type(result).__name__


T = t.TypeVar('T')


def save_as_html(
    serializer: t.Union[HtmlSerializer[T], WidgetSerializer[T]],
    file: t.Union[str, io.TextIOWrapper, None] = None,
    requirejs: bool = True,
    connected: bool = False,
    **kwargs
) -> t.Optional[str]:
    """Save a result to an HTML file.

    Parameters
    ----------
    serializer : Union[HtmlSerializer[T], WidgetSerializer[T]]
        serializer to prepare an output
    file : filename or file-like object
        The file to write the HTML output to. If None writes to output.html
    requirejs: bool , default: True
        whether to include requirejs library into output HTML or not
    connected: bool , default False
        indicates whether internet connection is available or not,
        if 'True' then CDN urls will be used to load javascript otherwise
        javascript libraries will be injected directly into HTML output.
        Set to 'False' to make results viewing possible when the internet
        connection is not available.

    Returns
    -------
    Optional[str] :
        name of newly create file
    """
    if file is None:
        file = 'output.html'
    if isinstance(file, str):
        file = create_new_file_name(file)

    if isinstance(serializer, WidgetSerializer):
        widget_to_html(
            serializer.serialize(**kwargs),
            html_out=file,
            title=get_result_name(serializer.value),
            requirejs=requirejs,
            connected=connected,
        )
    elif isinstance(serializer, HtmlSerializer):
        html = serializer.serialize(  # pylint: disable=redefined-outer-name
            full_html=True,
            **kwargs
        )
        if isinstance(file, str):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(html)
        elif isinstance(file, (io.TextIOWrapper, io.TextIOBase)):
            file.write(html)
        else:
            raise TypeError(f'Unsupported type of "file" parameter - {type(file)}')
    else:
        raise TypeError(f'Unsupported serializer type - {type(serializer)}')

    if isinstance(file, str):
        return file
