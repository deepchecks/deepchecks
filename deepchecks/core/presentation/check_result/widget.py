import typing as t

import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
from ipywidgets import HTML, VBox, DOMWidget, Widget

from deepchecks.core.check_result import CheckResult
from deepchecks.core.presentation.abc import WidgetSerializer
from . import html


__all__ = ['CheckResultSerializer']


class _WidgetSerializerImpl(html._HTMLSerializerImpl):

    def prepare_header(self, output_id: t.Optional[str] = None) -> HTML:
        return HTML(value=super().prepare_header(output_id))
    
    def prepare_summary(self) -> HTML:
        return HTML(value=super().prepare_summary())

    def prepare_conditions_table(
        self,
        max_info_len: int = 3000, 
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> HTML:
        widget = HTML(value=super().prepare_conditions_table(
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        ))
        return widget
    
    def prepare_additional_output(self, output_id: t.Optional[str] = None) -> VBox:
        return VBox(children=[
            HTML(value=it) if isinstance(it, str) else it
            for it in super().prepare_additional_output(output_id)
        ])
    
    @classmethod
    def handle_display_figure(cls, item: BaseFigure) -> go.FigureWidget:
        return go.FigureWidget(data=item)
    
    @classmethod
    def handle_display_string(cls, item: str) -> HTML:
        return HTML(value=super().handle_display_string(item))
    
    @classmethod
    def handle_display_dataframe(cls, item: pd.DataFrame) -> HTML:
        return HTML(value=super().handle_display_dataframe(item))
    
    @classmethod
    def handle_display_callable(cls, item: t.Callable) -> HTML:
        raise NotImplementedError()
        

class CheckResultSerializer(_WidgetSerializerImpl, WidgetSerializer[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value
    
    def serialize(
        self,
        output_id: t.Optional[str] = None,
        include: t.Optional[t.Sequence[html.CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        sections_to_include = html.verify_include_parameter(include)
        sections: t.List[Widget] = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(self.prepare_conditions_table(output_id=output_id))
        
        if 'additional-output' in sections_to_include:
            sections.append(self.prepare_additional_output(output_id))
        
        return normilize_widget_style(VBox(children=sections))


TDOMWidget = t.TypeVar('TDOMWidget', bound=DOMWidget)


def normilize_widget_style(w: TDOMWidget) -> TDOMWidget:
    return (
        w
        .add_class('rendered_html')
        .add_class('jp-RenderedHTMLCommon')
        .add_class('jp-RenderedHTML')
        .add_class('jp-OutputArea-output')
    )