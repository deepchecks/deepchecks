# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the check results classes."""
# pylint: disable=broad-except
import base64
import io
import traceback
import warnings
from typing import Any, Callable, List, Tuple, Union, TYPE_CHECKING

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import matplotlib
import pandas as pd
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
import plotly.io as pio
import plotly
from matplotlib import pyplot as plt
from IPython.display import display_html
from pandas.io.formats.style import Styler

from deepchecks.core.condition import Condition, ConditionCategory, ConditionResult
from deepchecks.core.display_pandas import dataframe_to_html, get_conditions_table
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.strings import create_new_file_name, get_docs_summary, widget_to_html
from deepchecks.utils.ipython import is_notebook
from deepchecks.utils.wandb_utils import set_wandb_run_state

# registers jsonpickle pandas extension for pandas support in the to_json function
jsonpickle_pd.register_handlers()

if TYPE_CHECKING:
    from deepchecks.core.checks import BaseCheck

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

__all__ = [
    'CheckResult',
    'CheckFailure',
]


def _save_all_open_figures():
    figs = [plt.figure(n) for n in plt.get_fignums()]
    images = []
    for fig in figs:
        bio = io.BytesIO()
        fig.savefig(bio, format='png')
        encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
        images.append(encoded)
        fig.clear()
    return images


_CONDITIONS_HEADER = '<h5>Conditions Summary</h5>'
_ADDITIONAL_OUTPUTS_HEADER = '<h5>Additional Outputs</h5>'


class CheckResult:
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Parameters
    ----------
    value : Any
        Value calculated by check. Can be used to decide if decidable check passed.
    display : List[Union[Callable, str, pd.DataFrame, Styler]] , default: None
        Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    header : str , default: None
        Header to be displayed in python notebook.
    """

    value: Any
    header: str
    display: List[Union[Callable, str, pd.DataFrame, Styler]]
    conditions_results: List[ConditionResult]
    check: 'BaseCheck'

    def __init__(self, value, header: str = None, display: Any = None):
        self.value = value
        self.header = header
        self.conditions_results = []

        if display is not None and not isinstance(display, List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Styler, Callable, BaseFigure)):
                raise DeepchecksValueError(f'Can\'t display item of type: {type(item)}')

    def display_check(self, unique_id: str = None, as_widget: bool = False,
                      show_additional_outputs=True):
        """Display the check result or return the display as widget.

        Parameters
        ----------
        unique_id : str
            The unique id given by the suite that displays the check.
        as_widget : bool
            Boolean that controls if to display the check regulary or if to return a widget.
        show_additional_outputs : bool
            Boolean that controls if to show additional outputs.
        Returns
        -------
        Widget
            Widget representation of the display if as_widget is True.
        """
        if as_widget:
            box = widgets.VBox()
            box.add_class('rendered_html')
            box_children = []
        check_html = ''
        if unique_id:
            check_html += f'<h4 id="{self.get_check_id(unique_id)}">{self.get_header()}</h4>'
        else:
            check_html += f'<h4>{self.get_header()}</h4>'
        if hasattr(self.check.__class__, '__doc__'):
            summary = get_docs_summary(self.check)
            check_html += f'<p>{summary}</p>'
        if self.conditions_results:
            check_html += _CONDITIONS_HEADER
            check_html += dataframe_to_html(get_conditions_table(self, unique_id))
        if show_additional_outputs:
            check_html += _ADDITIONAL_OUTPUTS_HEADER
            for item in self.display:
                if isinstance(item, (pd.DataFrame, Styler)):
                    check_html += dataframe_to_html(item)
                elif isinstance(item, str):
                    check_html += f'<div>{item}</div>'
                elif isinstance(item, BaseFigure):
                    if as_widget:
                        box_children.append(widgets.HTML(check_html))
                        box_children.append(go.FigureWidget(data=item))
                    else:
                        display_html(check_html, raw=True)
                        item.show()
                    check_html = ''
                elif callable(item):
                    try:
                        if as_widget:
                            plt_out = widgets.Output()
                            with plt_out:
                                item()
                                plt.show()
                            box_children.append(widgets.HTML(check_html))
                            box_children.append(plt_out)
                        else:
                            display_html(check_html, raw=True)
                            item()
                            plt.show()
                        check_html = ''
                    except Exception as exc:
                        check_html += f'Error in display {str(exc)}'
                else:
                    raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            check_html += '<p><b>&#x2713;</b> Nothing found</p>'
        if unique_id:
            check_html += f'<br><a href="#summary_{unique_id}" style="font-size: 14px">Go to top</a>'
        if as_widget:
            box_children.append(widgets.HTML(check_html))
            box.children = box_children
            return box
        display_html(check_html, raw=True)

    def _repr_html_(self, unique_id=None,
                    show_additional_outputs=True, requirejs: bool = False):
        """Return html representation of check result."""
        html_out = io.StringIO()
        self.save_as_html(html_out, unique_id=unique_id,
                          show_additional_outputs=show_additional_outputs, requirejs=requirejs)
        return html_out.getvalue()

    def save_as_html(self, file=None, unique_id=None,
                     show_additional_outputs=True, requirejs: bool = True):
        """Save output as html file.

        Parameters
        ----------
        file : filename or file-like object
            The file to write the HTML output to. If None writes to output.html
        requirejs: bool , default: True
            If to save with all javascript dependencies
        """
        if file is None:
            file = 'output.html'
        widgeted_output = self.display_check(unique_id=unique_id,
                                             show_additional_outputs=show_additional_outputs,
                                             as_widget=True)
        if isinstance(file, str):
            file = create_new_file_name(file, 'html')
        widget_to_html(widgeted_output, html_out=file, title=self.get_header(), requirejs=requirejs)

    def _display_to_json(self) -> List[Tuple[str, str]]:
        displays = []
        old_backend = matplotlib.get_backend()
        for item in self.display:
            if isinstance(item, Styler):
                displays.append(('dataframe', item.data.to_json(orient='records')))
            elif isinstance(item, pd.DataFrame):
                displays.append(('dataframe', item.to_json(orient='records')))
            elif isinstance(item, str):
                displays.append(('html', item))
            elif isinstance(item, BaseFigure):
                displays.append(('plotly', item.to_json()))
            elif callable(item):
                try:
                    matplotlib.use('Agg')
                    item()
                    displays.append(('plt', _save_all_open_figures()))
                except Exception:
                    displays.append(('plt', ''))
            else:
                matplotlib.use(old_backend)
                raise Exception(f'Unable to create json for item of type: {type(item)}')
        matplotlib.use(old_backend)
        return displays

    def to_wandb(self, dedicated_run: bool = True, **kwargs: Any):
        """Export check result to wandb.

        Parameters
        ----------
        dedicated_run : bool , default: None
            If to initiate and finish a new wandb run.
            If None it will be dedicated if wandb.run is None.
        kwargs: Keyword arguments to pass to wandb.init.
                Default project name is deepchecks.
                Default config is the check metadata (params, train/test/ name etc.).
        """
        check_metadata = self._get_metadata()
        section_suffix = check_metadata['header'] + '/'
        if isinstance(self.value, pd.DataFrame):
            value = self.value.to_json()
        elif isinstance(self.value, Styler):
            value = self.value.data.to_json()
        elif isinstance(self.value, np.ndarray):
            value = self.value.tolist()
        elif isinstance(self.value, (np.ndarray, np.generic)):
            value = un_numpy(self.value)
        else:
            value = jsonpickle.dumps(self.value, unpicklable=False)
        check_metadata['value'] = value
        dedicated_run = set_wandb_run_state(dedicated_run, check_metadata, **kwargs)
        if self.conditions_results:
            cond_df = get_conditions_table([self], icon_html=False)
            cond_table = wandb.Table(dataframe=cond_df.data, allow_mixed_types=True)
            wandb.log({f'{section_suffix}conditions_table': cond_table}, commit=False)
        table_i = 0
        plot_i = 0
        old_backend = matplotlib.get_backend()
        for item in self.display:
            if isinstance(item, Styler):
                wandb.log({f'{section_suffix}display_table_{table_i}':
                           wandb.Table(dataframe=item.data.reset_index(), allow_mixed_types=True)}, commit=False)
                table_i += 1
            elif isinstance(item, pd.DataFrame):
                wandb.log({f'{section_suffix}display_table_{table_i}':
                           wandb.Table(dataframe=item.reset_index(), allow_mixed_types=True)}, commit=False)
                table_i += 1
            elif isinstance(item, str):
                pass
            elif isinstance(item, BaseFigure):
                wandb.log({f'{section_suffix}plot_{plot_i}': wandb.Plotly(item)})
                plot_i += 1
            elif callable(item):
                try:
                    matplotlib.use('Agg')
                    item()
                    wandb.log({f'{section_suffix}plot_{plot_i}': plt})
                    plot_i += 1
                except Exception:
                    pass
            else:
                matplotlib.use(old_backend)
                raise Exception(f'Unable to process display for item of type: {type(item)}')

        matplotlib.use(old_backend)
        data = [check_metadata['header'],
                str(check_metadata['params']),
                check_metadata['summary'],
                value]
        final_table = wandb.Table(columns=['header', 'params', 'summary', 'value'])
        final_table.add_data(*data)
        wandb.log({f'{section_suffix}results': final_table}, commit=False)
        if dedicated_run:
            wandb.finish()

    def to_json(self, with_display: bool = True) -> str:
        """Return check result as json.

        Parameters
        ----------
        with_display : bool
            controls if to serialize display or not

        Returns
        -------
        str
            {'name': .., 'params': .., 'header': ..,
             'summary': .., 'conditions_table': .., 'value', 'display': ..}
        """
        result_json = self._get_metadata()
        if self.conditions_results:
            cond_df = get_conditions_table(self, icon_html=False)
            result_json['conditions_table'] = cond_df.data.to_json(orient='records')
        if isinstance(self.value, pd.DataFrame):
            result_json['value'] = self.value.to_json()
        elif isinstance(self.value, Styler):
            result_json['value'] = self.value.data.to_json()
        elif isinstance(self.value, np.ndarray):
            result_json['value'] = self.value.tolist()
        elif isinstance(self.value, (np.ndarray, np.generic)):
            result_json['value'] = un_numpy(self.value)
        else:
            result_json['value'] = self.value
        if with_display:
            display_json = self._display_to_json()
            result_json['display'] = display_json
        return jsonpickle.dumps(result_json, unpicklable=False)

    @staticmethod
    def display_from_json(json_data):
        """Display the check result from a json received from a to_json."""
        json_data = jsonpickle.loads(json_data)
        if json_data.get('display') is None:
            return
        header = json_data['header']
        summary = json_data['summary']
        display_html(f'<h4>{header}</h4>', raw=True)
        display_html(f'<p>{summary}</p>', raw=True)
        if json_data.get('conditions_table'):
            display_html(_CONDITIONS_HEADER, raw=True)
            conditions_table = pd.read_json(json_data['conditions_table'], orient='records')
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                display_html(dataframe_to_html(conditions_table.style.hide_index()), raw=True)
        display_html(_ADDITIONAL_OUTPUTS_HEADER, raw=True)
        for display_type, value in json_data['display']:
            if display_type == 'html':
                display_html(value, raw=True)
            elif display_type in ['conditions', 'dataframe']:
                df: pd.DataFrame = pd.read_json(value, orient='records')
                display_html(dataframe_to_html(df), raw=True)
            elif display_type == 'plotly':
                plotly_json = io.StringIO(value)
                plotly.io.read_json(plotly_json).show()
            elif display_type == 'plt':
                display_html(f'<img src=\'data:image/png;base64,{value}\'>', raw=True)
            else:
                raise ValueError(f'Unexpected type of display received: {display_type}')

    def _get_metadata(self, with_doc_link: bool = False):
        check_name = self.check.name()
        parameters = self.check.params(True)
        header = self.get_header()
        return {'name': check_name, 'params': parameters, 'header': header,
                'summary': get_docs_summary(self.check, with_doc_link=with_doc_link)}

    def _ipython_display_(self, unique_id=None, as_widget=False,
                          show_additional_outputs=True):
        check_widget = self.display_check(unique_id=unique_id, as_widget=as_widget,
                                          show_additional_outputs=show_additional_outputs)
        if as_widget:
            display_html(check_widget)

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return f'{self.get_header()}: {self.value}'

    def get_header(self) -> str:
        """Return header for display. if header was defined return it, else extract name of check class."""
        return self.header or self.check.name()

    def get_check_id(self, unique_id: str = '') -> str:
        """Return check id (used for href)."""
        header = self.get_header().replace(' ', '')
        return f'{header}_{unique_id}'

    def process_conditions(self) -> List[Condition]:
        """Process the conditions results from current result and check."""
        self.conditions_results = self.check.conditions_decision(self)

    def have_conditions(self) -> bool:
        """Return if this check has condition results."""
        return bool(self.conditions_results)

    def have_display(self) -> bool:
        """Return if this check has display."""
        return bool(self.display)

    def passed_conditions(self) -> bool:
        """Return if this check has no passing condition results."""
        return all((r.is_pass for r in self.conditions_results))

    @property
    def priority(self) -> int:
        """Return priority of the current result.

        This value is primarly used to determine suite output order.
        The logic is next:

        * if at least one condition did not pass and is of category 'FAIL', return 1.
        * if at least one condition did not pass and is of category 'WARN', return 2.
        * if check result do not have assigned conditions, return 3.
        * if all conditions passed, return 4.

        Returns
        -------
        int
            priority of the check result.
        """
        if not self.have_conditions:
            return 3

        for c in self.conditions_results:
            if c.is_pass is False and c.category == ConditionCategory.FAIL:
                return 1
            if c.is_pass is False and c.category == ConditionCategory.WARN:
                return 2

        return 4

    def show(self, show_additional_outputs=True, unique_id=None):
        """Display the check result.

        Parameters
        ----------
        show_additional_outputs : bool
            Boolean that controls if to show additional outputs.
        unique_id : str
            The unique id given by the suite that displays the check.
        """
        if is_notebook():
            self.display_check(unique_id=unique_id,
                               show_additional_outputs=show_additional_outputs)
        elif 'sphinx_gallery' in pio.renderers.default:
            html = self._repr_html_(unique_id=unique_id,
                                    show_additional_outputs=show_additional_outputs)

            class TempSphinx:
                def _repr_html_(self):
                    return html
            return TempSphinx()
        else:
            warnings.warn('You are running in a non-interactive python shell. in order to show result you have to use '
                          'an IPython shell (etc Jupyter)')


class CheckFailure:
    """Class which holds a check run exception.

    Parameters
    ----------
    check : BaseCheck
    exception : Exception
    header_suffix : str , default ``

    """

    def __init__(self, check: 'BaseCheck', exception: Exception, header_suffix: str = ''):
        self.check = check
        self.exception = exception
        self.header = check.name() + header_suffix

    def to_json(self, with_display: bool = True):
        """Return check failure as json.

        Parameters
        ----------
        with_display : bool
            controls if to serialize display or not

        Returns
        -------
        dict
            {'name': .., 'params': .., 'header': .., 'display': ..}
        """
        result_json = self._get_metadata()
        if with_display:
            result_json['display'] = [('html', f'<p style="color:red">{self.exception}</p>')]
        return jsonpickle.dumps(result_json, unpicklable=False)

    def to_wandb(self, dedicated_run: bool = True, **kwargs: Any):
        """Export check result to wandb.

        Parameters
        ----------
        dedicated_run : bool , default: None
            If to initiate and finish a new wandb run.
            If None it will be dedicated if wandb.run is None.
        kwargs: Keyword arguments to pass to wandb.init.
                Default project name is deepchecks.
                Default config is the check metadata (params, train/test/ name etc.).
        """
        check_metadata = self._get_metadata()
        section_suffix = check_metadata['header'] + '/'
        data = [check_metadata['header'],
                str(check_metadata['params']),
                check_metadata['summary'],
                str(self.exception)]
        check_metadata['value'] = str(self.exception)
        dedicated_run = set_wandb_run_state(dedicated_run, check_metadata, **kwargs)
        final_table = wandb.Table(columns=['header', 'params', 'summary', 'value'])
        final_table.add_data(*data)
        wandb.log({f'{section_suffix}results': final_table}, commit=False)
        if dedicated_run:
            wandb.finish()

    def _get_metadata(self, with_doc_link: bool = False):
        check_name = self.check.name()
        parameters = self.check.params(True)
        summary = get_docs_summary(self.check, with_doc_link=with_doc_link)
        return {'name': check_name, 'params': parameters, 'header': self.header, 'summary': summary}

    def __repr__(self):
        """Return string representation."""
        return self.header + ': ' + str(self.exception)

    def _ipython_display_(self):
        """Display the check failure."""
        check_html = f'<h4>{self.header}</h4>'
        if hasattr(self.check.__class__, '__doc__'):
            summary = get_docs_summary(self.check)
            check_html += f'<p>{summary}</p>'
        check_html += f'<p style="color:red">{self.exception}</p>'
        display_html(check_html, raw=True)

    def print_traceback(self):
        """Print the traceback of the failure."""
        tb_str = traceback.format_exception(etype=type(self.exception), value=self.exception,
                                            tb=self.exception.__traceback__)
        print(''.join(tb_str))
