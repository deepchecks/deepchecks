import typing as t
import textwrap
import warnings

import pandas as pd

from deepchecks.core import errors
from deepchecks.core.suite import SuiteResult
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import HtmlSerializer
from deepchecks.core.presentation.common import form_output_anchor
from deepchecks.core.presentation.common import aggregate_conditions
from deepchecks.core.presentation.common import Html
from deepchecks.core.presentation.dataframe import DataFramePresentation
from deepchecks.core.presentation.check_result.html import CheckResultSerializer as CheckResultHtmlSerializer 
from deepchecks.core.presentation.check_result.html import CheckResultSection


__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(HtmlSerializer[SuiteResult]):

    def __init__(self, value: SuiteResult, **kwargs):
        self.value = value
    
    def serialize(
        self,
        output_id: t.Optional[str] = None,
        **kwargs,
    ) -> str:
        summary = self.prepare_summary(**kwargs)
        conditions_table = self.prepare_conditions_table(**kwargs)
        failures = self.prepare_failures_list()
        
        results_with_conditions = self.prepare_results_with_condition_and_display(
            check_sections=['additional-output'],
            **kwargs
        )
        results_without_conditions = self.prepare_results_without_condition(
            check_sections=['additional-output'],
            **kwargs
        )
        sections = [
            summary,
            Html.bold_hr,
            conditions_table,
            Html.bold_hr,
            results_with_conditions,
            Html.bold_hr,
            results_without_conditions,
        ]

        if failures:
            sections.extend([Html.bold_hr, failures])
        
        if output_id:
            anchor = form_output_anchor(output_id)
            sections.append(f'<br><a href="{anchor}" style="font-size: 14px">Go to top</a>')
        
        return ''.join(sections)
    
    def prepare_prologue(self) -> str:
        long_prologue_version = "The suite is composed of various checks such as: {names}, etc..."
        short_prologue_version = "The suite is composed of the following checks: {names}."
        check_names = list(set(
            it.check.name() 
            for it in self.value.results
        ))
        return (
            long_prologue_version.format(names=', '.join(check_names[:3]))
            if len(check_names) > 3
            else short_prologue_version.format(names=', '.join(check_names))
        )
    
    def prepare_header(self, output_id: t.Optional[str] = None, **kwargs) -> str:
        idattr = f'id="{form_output_anchor(output_id)}"' if output_id else ''
        return f'<h1 {idattr}>{self.value.name}</h1>'
    
    def prepare_extra_info(self) -> str:
        if self.value.extra_info:
            extra_info = '<br>'.join(f'<div>{it}</div>' for it in self.value.extra_info)
            return f'<br>{extra_info}'
        else:
            return ''
    
    def prepare_summary(self, output_id: t.Optional[str] = None, **kwargs) -> str:
        header = self.prepare_header(output_id)
        prologue = self.prepare_prologue()
        extra_info = self.prepare_extra_info()

        suite_creation_example_link = (
            'https://docs.deepchecks.com/en/stable/examples/guides/create_a_custom_suite.html'
            '?utm_source=display_output&utm_medium=referral&utm_campaign=suite_link'
        )
        icons = textwrap.dedent("""
            <span style="color: green;display:inline-block">\U00002713</span> /
            <span style="color: red;display:inline-block">\U00002716</span> /
            <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span> /
            <span style="color: firebrick;font-weight:bold;display:inline-block">\U00002048</span>
        """)
        return textwrap.dedent(f"""
            {header}
            <p>
                {prologue}<br>
                Each check may contain conditions (which will result in pass / fail / warning / error
                , represented by {icons}) as well as other outputs such as plots or tables.<br>
                Suites, checks and conditions can all be modified. Read more about
                <a href={suite_creation_example_link} target="_blank">custom suites</a>.
            </p>
            {extra_info}
        """)
    
    def prepare_conditions_table(self, 
        output_id: t.Optional[str] = None,
        include_check_name: bool = False,
        **kwargs
    ):
        if not self.value.results_with_conditions:
            return '<p>No conditions defined on checks in the suite.</p>'
        
        table = DataFramePresentation(aggregate_conditions(
            self.value.results_with_conditions,
            output_id=output_id,
            include_check_name=include_check_name,
            max_info_len=300
        )).to_html()

        return f'<h2>Conditions Summary</h2>{table}'
    
    def prepare_results_with_condition_and_display(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> str:
        results_with_condition_and_display = [
            CheckResultHtmlSerializer(it).serialize(
                output_id=output_id, 
                include=check_sections,
                **kwargs
            )
            for it in self.value.results_with_conditions_and_display
        ]
        content = Html.light_hr.join(results_with_condition_and_display)
        return f'<h2>Check With Conditions Output</h2>{content}'

    def prepare_results_without_condition(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> str:
        results_without_conditions = [
            CheckResultHtmlSerializer(it).serialize(
                output_id=output_id, 
                include=check_sections,
                **kwargs
            )
            for it in self.value.results_without_conditions
        ]
        content = self.LIGHT_HR.join(results_without_conditions)
        return f'<h2>Check Without Conditions Output</h2>{content}'

    def prepare_failures_list(self) -> str:
        if not self.value.failed_or_without_display_results:
            return ''
        
        data = [] # type List[Tuple[check-header:str, message:str, priority:int]]
        
        for it in self.value.failed_or_without_display_results:
            if not isinstance(it, CheckFailure):
                data.append([it.get_header(), 'Nothing found', 2])
            else:
                error_types = (
                    errors.DatasetValidationError,
                    errors.ModelValidationError,
                    errors.DeepchecksProcessError,
                )
                message = (
                    str(it.exception) 
                    if isinstance(it.exception, error_types)
                    else f'{type(it.exception).__name__}: {str(it.exception)}'
                )
                data.append((it.header, message, 1 ))
        
        df = pd.DataFrame(data=data, columns=['Check', 'Reason', 'priority'])
        df.sort_values(by=['priority'], inplace=True)
        df.drop('priority', axis=1, inplace=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            table = DataFramePresentation(df.style.hide_index()).to_html()
            return f'<h2>Other Checks That Weren\'t Displayed</h2>\n{table}'
