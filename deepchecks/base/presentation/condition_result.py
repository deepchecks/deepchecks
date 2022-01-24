import typing as t

import pandas as pd
from pandas.io.formats.style import Styler
from ipywidgets import HTML
from ipywidgets import Widget

from deepchecks.base import ConditionResult
from deepchecks.base.presentation.abc import Presentation
from deepchecks.base.presentation.registry import default_presentation
from deepchecks.utils.strings import truncate_long_string


__all__ = ["ConditionResultPresentation"]


TConditionResult = t.Union[ConditionResult, t.List[ConditionResult]]


class ConditionResultPresentation(Presentation[TConditionResult]):

    def __init__(self, value: TConditionResult, **kwargs):
        assert isinstance(value, (ConditionResult, list))
        super().__init__(value, **kwargs)

    def as_html(
        self,
        *,
        check_header: t.Optional[str] = None,
        max_info_len: int = 3000,
        **kwargs
    ) -> str:
        condition_results = (
            self.value
            if isinstance(self.value, list)
            else [self.value]
        )

        table = []

        for condition_result in condition_results:
            if not check_header:
                table.append((
                    condition_result.get_icon(),
                    condition_result.name,
                    condition_result.details,
                    1 if condition_result.priority == 1 else 5,
                ))
            else:
                table.append((
                    condition_result.get_icon(),
                    check_header,
                    condition_result.name,
                    condition_result.details,
                    condition_result.priority,
                ))

        if not check_header:
            df = pd.DataFrame(
                data=table,
                columns=["Status", "Condition", "More Info", "sort"]
            )
        else:
            df = pd.DataFrame(
                data=table,
                columns=["Status", "Check", "Condition", "More Info", "sort"]
            )

        df.sort_values(by=['sort'], inplace=True)
        df.drop('sort', axis=1, inplace=True)
        df['More Info'] = df['More Info'].map(lambda x: truncate_long_string(x, max_info_len))

        presentation_type = default_presentation(Styler)
        return presentation_type(df.style.hide_index()).as_html()
    
    def as_widget(self, *, **kwargs) -> Widget:
        return HTML(value=self.as_html())