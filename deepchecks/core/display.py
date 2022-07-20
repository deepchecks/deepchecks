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
import html
import io
import pathlib
import sys
import typing as t
from multiprocessing import Process

import plotly.io as pio
from IPython.core.display import display, display_html
from ipywidgets import Widget

from deepchecks.core.serialization.abc import HTMLFormatter, HtmlSerializer, IPythonSerializer, WidgetSerializer
from deepchecks.utils.ipython import is_colab_env, is_databricks_env, is_kaggle_env, is_sagemaker_env
from deepchecks.utils.logger import get_logger
from deepchecks.utils.strings import create_new_file_name, get_random_string, widget_to_html, widget_to_html_string

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
    def ipython_serializer(self) -> IPythonSerializer[t.Any]:
        """Return IPythonSerializer instance."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def html_serializer(self) -> HtmlSerializer[t.Any]:
        """Return HtmlSerializer instance."""
        raise NotImplementedError()

    def show(
        self,
        as_widget: bool = True,
        unique_id: t.Optional[str] = None,
        **kwargs
    ) -> t.Optional[HTMLFormatter]:
        """Display result.

        Parameters
        ----------
        as_widget : bool, default True
            whether to display result with help of ipywidgets or not
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
        if 'sphinx_gallery' in pio.renderers.default:
            # TODO: why we need this? add comments
            html = widget_to_html_string(  # pylint: disable=redefined-outer-name
                self.widget_serializer.serialize(output_id=unique_id, **kwargs),
                title=get_result_name(self),
                requirejs=False,
                connected=True,
                full_html=False,
            )

            class TempSphinx:
                def _repr_html_(self):
                    return html

            return TempSphinx()

        if is_kaggle_env() or is_databricks_env() or is_sagemaker_env():
            self.show_in_iframe(as_widget=as_widget, unique_id=unique_id, **kwargs)
        elif is_colab_env() and as_widget is True:
            widget = self.widget_serializer.serialize(**kwargs)
            content = widget_to_html_string(widget, title=get_result_name(self))
            display_html(content, raw=True)
        elif is_colab_env() and as_widget is False:
            display(*self.ipython_serializer.serialize(**kwargs))
        elif as_widget is True:
            display_html(self.widget_serializer.serialize(
                output_id=unique_id,
                **kwargs
            ))
        else:
            display(*self.ipython_serializer.serialize(
                output_id=unique_id,
                **kwargs
            ))

    def show_in_iframe(
        self,
        as_widget: bool = True,
        unique_id: t.Optional[str] = None,
        connected: bool = False,
        **kwargs
    ):
        """Display result in an iframe.

        Parameters
        ----------
        as_widget : bool, default True
            whether to display result with help of ipywidgets or not
        unique_id : Optional[str], default None
            unique identifier of the result output
        connected: bool , default False
            indicates whether internet connection is available or not,
            if 'True' then CDN urls will be used to load javascript otherwise
            javascript libraries will be injected directly into HTML output.
            Set to 'False' to make results viewing possible when the internet
            connection is not available.
        **kwargs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method
        """
        output_id = unique_id or get_random_string(n=25)

        if is_colab_env() and as_widget is True:
            widget = self.widget_serializer.serialize(**kwargs)
            content = widget_to_html_string(widget, title=get_result_name(self), connected=True)
            display_html(content, raw=True)
        elif is_colab_env() and as_widget is False:
            display(*self.ipython_serializer.serialize(**kwargs))
        elif as_widget is True:
            widget = self.widget_serializer.serialize(output_id=output_id, is_for_iframe_with_srcdoc=True, **kwargs)
            content = widget_to_html_string(widget, title=get_result_name(self), connected=connected)
            display_html(iframe(srcdoc=content), raw=True)
        else:
            display_html(
                iframe(srcdoc=self.html_serializer.serialize(
                    output_id=output_id,
                    full_html=True,
                    include_requirejs=True,
                    include_plotlyjs=True,
                    is_for_iframe_with_srcdoc=True,
                    connected=connected,
                    **kwargs
                )),
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

        In this case, ipywidgets will not be used and plotly
        figures will be transformed into png images.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwrgs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method
        """
        display(*self.ipython_serializer.serialize(
            output_id=unique_id,
            plotly_to_image=True,
            **kwargs
        ))

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
    """Display suite result or check result in a new python gui window."""
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
            include_requirejs=requirejs,
            include_plotlyjs=True,
            connected=connected,
            **kwargs
        )
        if isinstance(file, str):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(html)
        elif isinstance(file, io.TextIOWrapper):
            file.write(html)
        else:
            raise TypeError(f'Unsupported type of "file" parameter - {type(file)}')
    else:
        raise TypeError(f'Unsupported serializer type - {type(serializer)}')

    if isinstance(file, str):
        return file


def iframe(
    *,
    id: t.Optional[str] = None,  # pylint: disable=redefined-builtin
    height: str = '600px',
    width: str = '100%',
    allow: str = 'fullscreen',
    frameborder: str = '0',
    with_fullscreen_btn: bool = True,
    **attributes
) -> str:
    """Return html iframe tag."""
    if id is None:
        id = f'deepchecks-result-iframe-{get_random_string()}'

    attributes = {
        'id': id,
        'height': height,
        'width': width,
        'allow': allow,
        'frameborder': frameborder,
        **attributes
    }
    attributes = {
        k: v
        for k, v
        in attributes.items()
        if v is not None
    }

    if 'srcdoc' in attributes:
        attributes['srcdoc'] = html.escape(attributes['srcdoc'])

    attributes = '\n'.join([
        f'{k}="{v}"'
        for k, v in attributes.items()
    ])

    if not with_fullscreen_btn:
        return f'<iframe {attributes}></iframe>'

    fullscreen_script = (
        f"document.querySelector('#{id}').requestFullscreen();"
    )
    return f"""
        <div style="display: flex; justify-content: flex-end; padding: 1rem 2rem 1rem 2rem;">
            <button onclick="{fullscreen_script}" >
                Full Screen
            </button>
        </div>
        <iframe allowfullscreen {attributes}></iframe>
    """
