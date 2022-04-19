import typing as t
import warnings
import json

import pandas as pd
import numpy as np
from ipywidgets import DOMWidget
from jsonpickle.pickler import Pickler
from pandas.io.formats.style import Styler

from deepchecks.utils.strings import get_ellipsis
from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import BaseCheck
from deepchecks.utils.dataframes import un_numpy


__all__ = [
    'aggregate_conditions', 
    'form_output_anchor', 
    'form_check_id', 
    'Html',
    'normalize_widget_style',
    'normilize_value',
    'pretify'
]


# class CustomNotebookRenderer(plotly_renderes.NotebookRenderer):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.connected = False
#         self._is_activated = False

#     def activate(self):
#         if self._is_activated is False:
#             super().activate()
#             self._is_activated = True

#     @property
#     def is_plotly_activated(self):
#         return self._is_activated


class Html:
    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'
    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 4px;">'


def form_output_anchor(output_id: str) -> str:
    return f'#summary_{output_id}'


def form_check_id(check: BaseCheck, output_id: str) -> str:
    check_name = type(check).__name__
    return f'{check_name}_{output_id}'


TDOMWidget = t.TypeVar('TDOMWidget', bound=DOMWidget)


def normalize_widget_style(w: TDOMWidget) -> TDOMWidget:
    return (
        w
        .add_class('rendered_html')
        .add_class('jp-RenderedHTMLCommon')
        .add_class('jp-RenderedHTML')
        .add_class('jp-OutputArea-output')
    )


def pretify(
    data: t.Union[t.List[t.Any], t.Dict[t.Any, t.Any]],
    indent: int = 3
) -> str:
    default = lambda it: repr(it)
    return json.dumps(data, indent=indent, default=default)


def normilize_value(value: object) -> t.Any:
    """Takes an object and returns a JSON-safe representation of it.

    Parameters
    ----------
    value : object
        value to normilize
    
    Returns
    -------
    Any of the basic builtin datatypes
    """
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient='records')
    elif isinstance(value, Styler):
        return value.data.to_json(orient='records')
    elif isinstance(value, (np.generic, np.ndarray)):
        return un_numpy(value)
    else:
        return Pickler(unpicklable=False).flatten(value)


def aggregate_conditions(
    check_results: t.Union['CheckResult', t.List['CheckResult']],
    max_info_len: int = 3000, 
    include_icon: bool = True,
    include_check_name: bool = False,
    output_id: t.Optional[str] = None,
) -> pd.DataFrame:
    """Return the conditions table as DataFrame.

    Parameters
    ----------
    check_results : Union['CheckResult', List['CheckResult']]
        check results to show conditions of.
    max_info_len : int
        max length of the additional info.
    include_icon : bool , default: True
        if to show the html condition result icon or the enum
    include_check_name : bool, default False
        whether to include check name into dataframe or not
    output_id : str
        the unique id to append for the check names to create links (won't create links if None/empty).
    
    Returns
    -------
    pd.Dataframe:
        the condition table.
    """
    check_results = [check_results] if isinstance(check_results, CheckResult) else check_results
    data = []
    
    for check_result in check_results:
        for cond_result in check_result.conditions_results:
            priority = cond_result.priority
            icon = cond_result.get_icon() if include_icon else cond_result.category.value
            check_header = check_result.get_header()
            
            if output_id and check_result.have_display():
                link = f'<a href=#{check_result.get_check_id(output_id)}>{check_header}</a>'
            else:
                link = check_header
                # if it has no display show on bottom for the category (lower priority)
                priority += 0.1
            
            data.append([
                icon, link, cond_result.name, cond_result.details, priority
            ])

    df = pd.DataFrame(
        data=data,
        columns=['Status', 'Check', 'Condition', 'More Info', 'sort']
    )
    
    df.sort_values(by=['sort'], inplace=True)
    df.drop('sort', axis=1, inplace=True)
    
    if include_check_name is False:
        df.drop('Check', axis=1, inplace=True)
    
    df['More Info'] = df['More Info'].map(lambda x: get_ellipsis(x, max_info_len))
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        return df.style.hide_index()