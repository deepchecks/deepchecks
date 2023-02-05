import io
import typing as t

from ipywidgets import HTML, VBox, Widget

from deepchecks.core.display import DisplayableResult
from deepchecks.core.serialization.abc import HtmlSerializer, IPythonSerializer, WidgetSerializer
from deepchecks.core.serialization.common import normalize_widget_style
from deepchecks.utils.strings import create_new_file_name


class HtmlDisplayableResult(DisplayableResult):
    """Class which accepts html string and support displaying it in different environments."""

    def __init__(self, html: str):
        self.html = html

    @property
    def widget_serializer(self) -> WidgetSerializer[t.Any]:
        class _WidgetSerializer(WidgetSerializer[t.Any]):
            def serialize(self, **kwargs) -> Widget:  # pylint: disable=unused-argument
                return normalize_widget_style(VBox(children=[HTML(self.value)]))

        return _WidgetSerializer(self.html)

    @property
    def ipython_serializer(self) -> IPythonSerializer[t.Any]:
        class _IPythonSerializer(IPythonSerializer[t.Any]):
            def serialize(self, **kwargs) -> t.Any:  # pylint: disable=unused-argument
                return HTML(self.value)

        return _IPythonSerializer(self.html)

    @property
    def html_serializer(self) -> HtmlSerializer[t.Any]:
        class _HtmlSerializer(HtmlSerializer[t.Any]):
            def serialize(self, **kwargs) -> str:  # pylint: disable=unused-argument
                return self.value

        return _HtmlSerializer(self.html)

    def to_widget(self, **kwargs) -> Widget:
        return self.widget_serializer.serialize(**kwargs)

    def to_json(self, **kwargs):
        raise NotImplementedError()

    def to_wandb(self, **kwargs):
        raise NotImplementedError()

    def save_as_html(self, file: t.Union[str, io.TextIOWrapper, None] = None, **kwargs) -> t.Optional[str]:
        if file is None:
            file = 'output.html'
        if isinstance(file, str):
            file = create_new_file_name(file)

        if isinstance(file, str):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.html)
        elif isinstance(file, io.TextIOWrapper):
            file.write(self.html)

        return file
