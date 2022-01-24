import typing as t
import warnings

import pandas as pd
from pandas.io.formats.style import Styler
from ipywidgets import Widget
from ipywidgets import HTML

from .abc import Presentation


__all__ = ["DataFramePresentation"]


class DataFramePresentation(Presentation[t.Union[pd.DataFrame, Styler]]):
    
    def __init__(self, value: t.Union[pd.DataFrame, Styler], **kwargs):
        assert isinstance(value, (pd.DataFrame, Styler))
        super().__init__(value, **kwargs)
    
    def as_html(self, *, **kwargs) -> str:
        try:
            styler = (
                self.value.style 
                if isinstance(self.value, pd.DataFrame) 
                else self.value
            )
        except ValueError:
            # Dataframe with Multi-index or non unique indices does not have a style
            # attribute, hence we need to display as a regular pd html format.
            if isinstance(self.value, pd.DataFrame):
                return self.value.to_html()
            else:
                raise
            
        # We are using deprecated pandas method and want to hide the warning about that
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            styler.set_precision(2)

        table_css_props = [
            ('text-align', 'left'),  # Align everything to the left
            ('white-space', 'pre-wrap')  # Define how to handle white space characters (like \n)
        ]
        
        styler.set_table_styles([dict(selector='table,thead,tbody,th,td', props=table_css_props)])
        return styler.render()
        
    def as_widget(self, *, **kwargs) -> Widget:
        return HTML(value=self.as_html())
