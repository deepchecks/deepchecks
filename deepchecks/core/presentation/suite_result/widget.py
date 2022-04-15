
import typing as t
import warnings

import pandas as pd
from ipywidgets import Widget, HTML, VBox, Tab

from deepchecks.utils.strings import get_docs_summary
from deepchecks.core.suite import SuiteResult
from deepchecks.core.presentation.abc import WidgetSerializer
from deepchecks.core.presentation.common import Html as CommonHtml
from deepchecks.core.presentation.common import normalize_widget_style
from deepchecks.core.presentation.check_result.html import CheckResultSection
from deepchecks.core.presentation.check_result.widget import CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.core.presentation.dataframe import DataFramePresentation

from . import html


__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(WidgetSerializer[SuiteResult]):
    
    def __init__(self, value: SuiteResult, **kwargs):
        super().__init__(**{'value': value, **kwargs},)
        self.value = value
        self._html_serializer = html.SuiteResultSerializer(self.value)
    
    def serialize(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> VBox:
        tab = Tab()
        tab.set_title(0, 'Checks With Conditions')
        tab.set_title(1, 'Checks Without Conditions')
        tab.set_title(2, 'Checks Without Output')
        
        tab.children = [
            self.prepare_results_with_condition_and_display(
                output_id=output_id, **kwargs
            ),
            self.prepare_results_without_condition(
                output_id=output_id,
                check_sections=['additional-output'],
                **kwargs
            ), 
            self.prepare_failures_list()
        ]

        style = '<style>.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {flex: 0 1 auto}</style>'

        return VBox(children=[
            HTML(value=style),
            self.prepare_summary(output_id=output_id, **kwargs),
            tab
        ])

    def prepare_summary(
        self, 
        output_id: t.Optional[str] = None, 
        **kwargs
    ) -> HTML:
        return HTML(value=self._html_serializer.prepare_summary(
            output_id, 
            **kwargs
        ))

    def prepare_conditions_table(
        self, 
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> HTML:
        return normalize_widget_style(HTML(value=self._html_serializer.prepare_conditions_table(
            output_id,
            include_check_name=True,
            **kwargs
        )))
    
    def prepare_failures_list(self) -> HTML:
        return normalize_widget_style(HTML(
            value=self._html_serializer.prepare_failures_list()
        ))

    def prepare_results_without_condition(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        results_without_conditions = [
            CheckResultWidgetSerializer(it).serialize(
                output_id=output_id, 
                include=check_sections,
                **kwargs
            )
            for it in self.value.results_without_conditions
        ]
        return normalize_widget_style(VBox(children=[
            HTML(value='<h2>Check Without Conditions Output</h2>'),
            self.prepare_navigation_for_unconditioned_results(output_id),
            HTML(value=CommonHtml.light_hr),
            *join(results_without_conditions, HTML(value=CommonHtml.light_hr))
        ]))

    def prepare_results_with_condition_and_display(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        results_with_condition_and_display = [
            CheckResultWidgetSerializer(it).serialize(
                output_id=output_id,
                include=check_sections,
                **kwargs
            )
            for it in self.value.results_with_conditions_and_display
        ]
        
        return normalize_widget_style(VBox(children=[
            self.prepare_conditions_table(),
            HTML(value='<h2>Check With Conditions Output</h2>'),
            *join(results_with_condition_and_display, HTML(value=CommonHtml.light_hr))
        ]))
        
    def prepare_navigation_for_unconditioned_results(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> Widget:
        data = []
        
        for check_result in self.value.results_without_conditions:
            check_header = check_result.get_header()

            if output_id:
                anchor = f'href="#{check_result.get_check_id(output_id)}'
                header = f'<a href="{anchor}">{check_header}</a>'
            else:
                header = check_header

            summary = get_docs_summary(check_result.check)
            data.append([header, summary])

        df = pd.DataFrame(
            data=data,
            columns=['Check', 'Summary']
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            return DataFramePresentation.to_widget(df.style.hide_index())


A = t.TypeVar('A')
B = t.TypeVar('B')


def join(l: t.List[A], item: B) -> t.Iterator[t.Union[A, B]]:
    list_len = len(l) - 1
    for index, el in enumerate(l):
        yield el
        if index != list_len:
            yield item
