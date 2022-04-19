from ipywidgets import VBox, HTML

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import WidgetSerializer

from . import html


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WidgetSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value
        self._html_serializer = html.CheckFailureSerializer(self.value)
    
    def serialize(self, **kwargs) -> VBox:
        return VBox(children=(
            self.prepare_header(),
            self.prepare_summary(),
            self.prepare_error_message()
        ))

    def prepare_header(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_header())
    
    def prepare_summary(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_summary())
    
    def prepare_error_message(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_error_message())