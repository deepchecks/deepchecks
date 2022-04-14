from ipywidgets import VBox
from deepchecks.core.check_result import CheckResult
from deepchecks.core.presentation.abc import Presentation
from . import html, widget


__all__ = ['CheckResultPresentation']


class CheckResultPresentation(Presentation[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value
        self._html_serializer = html.CheckResultSerializer(self.value)
        self._widget_serializer = widget.CheckResultSerializer(self.value)

    def to_html(self, **kwargs) -> str:
        return self._html_serializer.serialize(**kwargs)

    def to_widget(self, **kwargs) -> VBox:
        return self._widget_serializer.serialize(**kwargs)