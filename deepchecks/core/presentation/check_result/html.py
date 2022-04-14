import typing as t

import pandas as pd
import plotly.io as pio
# import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from typing_extensions import Literal

from deepchecks.core.check_result import CheckResult
from deepchecks.utils.strings import get_docs_summary
from deepchecks.core.presentation.abc import HtmlSerializer
from deepchecks.core.presentation.dataframe import DataFramePresentation
from deepchecks.core.presentation.common import aggregate_conditions, form_output_anchor, form_check_id


__all__ = ['CheckResultSerializer']


CheckResultSection = t.Union[
    Literal['condition-table'],
    Literal['additional-output'],
]


class _HTMLSerializerImpl:
    value: CheckResult
    
    def prepare_header(self, output_id: t.Optional[str] = None) -> str:
        header = self.value.get_header()
        header = f'<b>{header}</b>'
        if output_id is not None:
            check_id = form_check_id(self.value.check, output_id)
            return f'<h4 id="{check_id}">{header}</h4>'
        else:
            return f'<h4>{header}</h4>'
    
    def prepare_summary(self) -> str:
        summary = get_docs_summary(self.value.check)
        return f'<p>{summary}</p>'
    
    def prepare_conditions_table(
        self,
        max_info_len: int = 3000, 
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> str:
        table = DataFramePresentation(aggregate_conditions(
            self.value, 
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        )).to_html()
        return f'<h5>Conditions Summary</h5>{table}'
    
    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None
    ) -> t.List[str]:
        output = ['<h5><b>Additional Outputs</b></h5>']

        for item in self.value.display:
            if isinstance(item, str):
                output.append(self.handle_display_string(item))

            elif isinstance(item, (pd.DataFrame, Styler)):
                output.append(self.handle_display_dataframe(item))

            elif isinstance(item, BaseFigure):
                output.append(self.handle_display_figure(item))

            elif callable(item):
                output.append(self.handle_display_callable(item))

            else:
                raise TypeError(f'Unable to display item of type: {type(item)}')

        if len(self.value.display) == 0:
            output.append('<p><b>&#x2713;</b>Nothing to display</p>')

        if output_id is not None:
            href = form_output_anchor(output_id)
            output.append(f'<br><a href="{href}" style="font-size: 14px">Go to top</a>')

        return output
    
    @classmethod
    def handle_display_string(cls, item):
        return f'<div>{item}</div>'
    
    @classmethod
    def handle_display_dataframe(cls, item):
        return DataFramePresentation(item).to_html()
    
    @classmethod
    def handle_display_callable(cls, item):
        raise NotImplementedError()

    @classmethod
    def handle_display_figure(cls, item):
        bundle = pio.renderers['notebook'].to_mimebundle(item)  # dict structure: {"text/html": value}
        return bundle['text/html']


class CheckResultSerializer(_HTMLSerializerImpl, HtmlSerializer[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        include: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> str:
        sections_to_include = verify_include_parameter(include)
        sections = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(''.join(self.prepare_conditions_table(output_id=output_id)))

        if 'additional-output' in sections_to_include:
            sections.append(''.join(self.prepare_additional_output(output_id)))
        
        return ''.join(sections)


def verify_include_parameter(
    include: t.Optional[t.Sequence[CheckResultSection]] = None
) -> t.Set[CheckResultSection]:
    sections = t.cast(
        t.Set[CheckResultSection],
        {'condition-table', 'additional-output'}
    )

    if include is None:
        sections_to_include = sections
    elif len(include) == 0:
        raise ValueError('include parameter cannot be empty')
    else:
        sections_to_include = set(include)

    if len(sections_to_include.difference(sections)) > 0:
        raise ValueError(
            'include parameter must contain '
            'Union[Literal["condition-table"], Literal["additional-output"]]'
        )

    return sections_to_include
