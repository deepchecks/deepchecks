import typing as t

import pandas as pd
from ipywidgets import HTML
from ipywidgets import Widget

from deepchecks.base import SuiteResult
from deepchecks.base.presentation.abc import Presentation


__all__ = ["SuiteResultPresentation"]


class SuiteResultPresentation(Presentation[SuiteResult]):
    
    def __init__(self, value: SuiteResult, **kwargs):
        assert isinstance(value, SuiteResult)
        super().__init__(value, **kwargs)
    
    def as_html(self, *, **kwargs) -> str:
        return super().as_html(**kwargs)

