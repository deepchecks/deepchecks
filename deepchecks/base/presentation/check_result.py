import typing as t

import pandas as pd
from pandas.io.formats.style import Styler
from ipywidgets import Widget
from ipywidgets import HTML
from plotly.basedatatypes import BaseFigure

from deepchecks.base import CheckResult
from deepchecks.base import ConditionResult
from deepchecks.base.presentation.abc import Presentation
from deepchecks.base.presentation.registry import default_presentation
from deepchecks.utils.strings import get_docs_summary


__all__ = ["CheckResultPresentation"]


class CheckResultPresentation(Presentation[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        assert isinstance(value, CheckResult)
        super().__init__(value, **kwargs)

    def as_html(
        self, 
        *, 
        suite_id: t.Optional[str] = None,
        additional_output: bool = False,
        **kwargs
    ) -> str:
        check = self.value.check
        check_name = type(check).__name__

        if suite_id:
            check_id = f"{check_name}_{suite_id}"
            header = f"<h4 id='{check_id}'>{self.value.get_header()}</h4>"
            check_reference = f"<a href='#{check_id}'>{self.value.get_header()}</a>"
            suite_reference = f"<br><a href='#summary_{suite_id}' style='font-size: 14px'>Go to top</a>"
        else:
            check_id = None
            header = f"<h4>{self.value.get_header()}</h4>"
            check_reference = None
            suite_reference = ""
        
        presentation_type = default_presentation(t.List[ConditionResult])
        conditions_presentation = presentation_type(self.value.conditions_results)
        
        output = [
            header,
            f"<p>{get_docs_summary(check)}</p>",
            f"<h5>Conditions Summary</h5>",
            conditions_presentation.as_html(check_header=check_reference),
        ]

        if additional_output is False:
            output.append(suite_reference)
            return "".join(output)
        
        if not self.value.display:
            output.append("<p><b>&#x2713;</b> Nothing found</p>")
            output.append(suite_reference)
            return "".join(output)
        
        output.append("<h5>Additional Outputs</h5>")
        
        for item in self.value.display:
            if isinstance(item, (pd.DataFrame, Styler)):
                presentation_type = default_presentation(t.Union[pd.DataFrame, Styler])
                output.append(presentation_type(item).as_html())
            elif isinstance(item, str):
                output.append(f"<p>{item}</p>")
            elif isinstance(item, BaseFigure):
                output.append(item.to_html(
                    full_html=False, 
                    include_plotlyjs="cdn"
                ))
            elif callable(item):
                # TODO:
                output.append("TODO: add handler for the callables")
            else:
                raise TypeError(f"Cannot handle display item of type - {type(item)}")
        
        output.append(suite_reference)
        return "".join(output)

    def as_widget(self, *, **kwargs) -> Widget:
        return HTML(value=self.as_html(**kwargs))