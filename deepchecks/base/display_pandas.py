"""Handle displays of pandas objects."""
from typing import Union

from IPython.core.display import display_html
import pandas as pd
__all__ = ['display_dataframe', 'dataframe_to_html']

from pandas.io.formats.style import Styler


def display_dataframe(df: Union[pd.DataFrame, Styler], hide_index=False):
    """Display in IPython given dataframe.

    Args:
        df (pd.DataFrame): Dataframe to display
        hide_index (bool): Whether to hide or not the dataframe index.
    """
    display_html(dataframe_to_html(df, hide_index), raw=True)


def dataframe_to_html(df: Union[pd.DataFrame, Styler], hide_index=False):
    """Convert dataframe to html.

    Args:
        df (pd.DataFrame): Dataframe to convert.
        hide_index (bool): Whether to hide or not the dataframe index.
    """
    # Align everything to the left
    try:
        if isinstance(df, pd.DataFrame):
            df_styler = df.style
        else:
            df_styler = df
        df_styler.set_table_styles([dict(selector='table,thead,tbody,th,td', props=[('text-align', 'left')])])
        df_styler.format(precision=2)
        if hide_index:
            df_styler.hide_index()
        return df_styler.render()
    # Because of MLC-154. Dataframe with Multi-index or non unique indices does not have a style
    # attribute, hence we need to display as a regular pd html format.
    except ValueError:
        return df.to_html()
