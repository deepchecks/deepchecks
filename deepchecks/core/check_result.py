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
# pylint: disable=broad-except,import-outside-toplevel,unused-argument
import io
import traceback
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import pandas as pd
import plotly.io as pio
from IPython.display import display_html
from ipywidgets import Widget
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure

from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.serialization.check_failure.html import \
    CheckFailureSerializer as CheckFailureHtmlSerializer
from deepchecks.core.serialization.check_failure.json import \
    CheckFailureSerializer as CheckFailureJsonSerializer
from deepchecks.core.serialization.check_result.html import \
    CheckResultSerializer as CheckResultHtmlSerializer
from deepchecks.core.serialization.check_result.json import \
    CheckResultSerializer as CheckResultJsonSerializer
from deepchecks.core.serialization.check_result.widget import \
    CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.utils.ipython import (is_colab_env, is_kaggle_env, is_notebook,
                                      is_widgets_use_possible)
from deepchecks.utils.strings import create_new_file_name, widget_to_html
from deepchecks.utils.wandb_utils import set_wandb_run_state

# registers jsonpickle pandas extension for pandas support in the to_json function
jsonpickle_pd.register_handlers()


if TYPE_CHECKING:
    from deepchecks.core.checks import BaseCheck


__all__ = ['CheckResult', 'CheckFailure']


TDisplayCallable = Callable[[], None]
TDisplayItem = Union[str, pd.DataFrame, Styler, BaseFigure, TDisplayCallable]


class BaseCheckResult:
    """Generic class for any check output, contains some basic functions."""

    check: Optional['BaseCheck']
    header: Optional[str]

    @staticmethod
    def from_json(json_dict: Union[str, Dict]) -> 'BaseCheckResult':
        """Convert a json object that was returned from CheckResult.to_json or CheckFailure.to_json.

        Parameters
        ----------
        json_data: Union[str, Dict]
            Json data

        Returns
        -------
        BaseCheckResult
            A check output object.
        """
        from deepchecks.core.check_json import (CheckFailureJson,
                                                CheckResultJson)

        if isinstance(json_dict, str):
            json_dict = jsonpickle.loads(json_dict)
        check_type = json_dict['type']
        if check_type == 'CheckFailure':
            return CheckFailureJson(json_dict)
        elif check_type == 'CheckResult':
            return CheckResultJson(json_dict)
        else:
            raise ValueError('Excpected json object to be one of [CheckFailure, CheckResult] but recievied: ' + type)

    def get_header(self) -> str:
        """Return header for display. if header was defined return it, else extract name of check class."""
        return self.header or self.check.name()

    def get_metadata(self, with_doc_link: bool = False) -> Dict:
        """Return the related check metadata."""
        return {'header': self.get_header(), **self.check.metadata(with_doc_link=with_doc_link)}

    def get_check_id(self, unique_id: str = '') -> str:
        """Return check id (used for href)."""
        header = self.get_header().replace(' ', '')
        return f'{header}_{unique_id}'


class CheckResult(BaseCheckResult):
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Parameters
    ----------
    value : Any
        Value calculated by check. Can be used to decide if decidable check passed.
    display : List[Union[Callable, str, pd.DataFrame, Styler, BaseFigure]] , default: None
        Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    header : str , default: None
        Header to be displayed in python notebook.
    """

    value: Any
    header: Optional[str]
    display: List[TDisplayItem]
    conditions_results: List[ConditionResult]

    def __init__(
        self,
        value,
        header: Optional[str] = None,
        display: Optional[List[TDisplayItem]] = None
    ):
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

    def to_widget(
        self,
        unique_id: Optional[str] = None,
        show_additional_outputs: bool = True
    ) -> Widget:
        """Return CheckResult as a ipywidgets.Widget instance.

        Parameters
        ----------
        unique_id : str
            The unique id given by the suite that displays the check.
        show_additional_outputs : bool
            Boolean that controls if to show additional outputs.

        Returns
        -------
        Widget
        """
        check_sections = (
            ['condition-table', 'additional-output']
            if show_additional_outputs is True
            else ['condition-table']
        )
        return CheckResultWidgetSerializer(self).serialize(
            output_id=unique_id,
            check_sections=check_sections  # type: ignore
        )

    def display_check(
        self,
        unique_id: Optional[str] = None,
        as_widget: bool = False,
        show_additional_outputs: bool = True,
        full_html: bool = False
    ) -> Optional[Widget]:
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
        if as_widget is True:
            return self.to_widget(
                unique_id=unique_id,
                show_additional_outputs=show_additional_outputs
            )
        else:
            check_sections = (
                ['condition-table', 'additional-output']
                if show_additional_outputs
                else ['condition-table']
            )
            display_html(
                CheckResultHtmlSerializer(self).serialize(
                    output_id=unique_id,
                    full_html=full_html,
                    check_sections=check_sections  # type: ignore
                ),
                raw=True,
            )

    def _repr_html_(
        self,
        unique_id: Optional[str] = None,
        show_additional_outputs: bool = True,
        requirejs: bool = False
    ) -> str:
        """Return html representation of check result."""
        html_out = io.StringIO()
        self.save_as_html(
            html_out,
            unique_id=unique_id,
            show_additional_outputs=show_additional_outputs,
            requirejs=requirejs
        )
        return html_out.getvalue()

    def save_as_html(
        self,
        file: Union[str, io.TextIOWrapper, None] = None,
        unique_id: Optional[str] = None,
        show_additional_outputs: bool = True,
        requirejs: bool = True
    ):
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
        if isinstance(file, str):
            file = create_new_file_name(file)

        widget_to_html(
            self.to_widget(
                unique_id=unique_id,
                show_additional_outputs=show_additional_outputs,
            ),
            html_out=file,
            title=self.get_header(),
            requirejs=requirejs
        )

    def to_wandb(
        self,
        dedicated_run: bool = True,
        **kwargs: Any
    ):
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
        # NOTE: Wandb is not a default dependency
        # user should install it manually therefore we are
        # doing import within method to prevent premature ImportError
        try:
            import wandb
            from deepchecks.core.serialization.check_result.wandb import \
                CheckResultSerializer as WandbSerializer
        except ImportError as error:
            raise ImportError(
                'Wandb serializer requires the wandb python package. '
                'To get it, run "pip install wandb".'
            ) from error
        else:
            dedicated_run = set_wandb_run_state(
                dedicated_run,
                {'header': self.get_header(), **self.check.metadata()},
                **kwargs
            )
            wandb.log(WandbSerializer(self).serialize())
            if dedicated_run:  # TODO: create context manager for this
                wandb.finish()

    def to_json(self, with_display: bool = True) -> str:
        """Return check result as json.

        Returned JSON string will have next structure:

        >>    class CheckResultMetadata(TypedDict):
        >>        type: str
        >>        check: CheckMetadata
        >>        value: Any
        >>        header: str
        >>        conditions_results: List[Dict[Any, Any]]
        >>        display: List[Dict[str, Any]]

        >>    class CheckMetadata(TypedDict):
        >>        name: str
        >>        params: Dict[Any, Any]
        >>        summary: str

        Parameters
        ----------
        with_display : bool
            controls if to serialize display or not

        Returns
        -------
        str
        """
        # TODO: not sure if the `with_display` parameter is needed
        # add deprecation warning if it is not needed
        return jsonpickle.dumps(
            CheckResultJsonSerializer(self).serialize(with_display=with_display),
            unpicklable=False
        )

    def _ipython_display_(
        self,
        unique_id: Optional[str] = None,
        as_widget: bool = True,
        show_additional_outputs: bool = True
    ):
        as_widget = is_widgets_use_possible() and as_widget
        check_sections = (
            ['condition-table', 'additional-output']
            if show_additional_outputs
            else ['condition-table']
        )

        if as_widget is True:
            display_html(CheckResultWidgetSerializer(self).serialize(
                output_id=unique_id,
                check_sections=check_sections  # type: ignore
            ))
        else:
            is_colab = is_colab_env()
            is_kaggle = is_kaggle_env()
            display_html(
                CheckResultHtmlSerializer(self).serialize(
                    output_id=unique_id if not is_colab else None,
                    full_html=is_colab,
                    include_requirejs=is_kaggle,
                    connected=not is_kaggle
                ),
                raw=True
            )

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return f'{self.get_header()}: {self.value}'

    def process_conditions(self):
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
            self.display_check(
                unique_id=unique_id,
                show_additional_outputs=show_additional_outputs
            )
        elif 'sphinx_gallery' in pio.renderers.default:
            html = self._repr_html_(
                unique_id=unique_id,
                show_additional_outputs=show_additional_outputs
            )

            class TempSphinx:
                def _repr_html_(self):
                    return html

            return TempSphinx()
        else:
            warnings.warn(
                'You are running in a non-interactive python shell. '
                'In order to show result you have to use '
                'an IPython shell (etc Jupyter)'
            )


class CheckFailure(BaseCheckResult):
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

        Returned JSON string will have next structure:

        >>    class CheckFailureMetadata(TypedDict):
        >>        check: CheckMetadata
        >>        header: str
        >>        display: List[Dict[str, str]]

        >>    class CheckMetadata(TypedDict):
        >>        type: str
        >>        name: str
        >>        params: Dict[Any, Any]
        >>        summary: str

        Parameters
        ----------
        with_display : bool
            controls if to serialize display or not

        Returns
        -------
        str
        """
        # TODO: not sure if the `with_display` parameter is needed
        # add deprecation warning if it is not needed
        return jsonpickle.dumps(
            CheckFailureJsonSerializer(self).serialize(),
            unpicklable=False
        )

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
        # NOTE: Wandb is not a default dependency
        # user should install it manually therefore we are
        # doing import within method to prevent premature ImportError
        try:
            import wandb
            from deepchecks.core.serialization.check_failure.wandb import \
                CheckFailureSerializer as WandbSerializer
        except ImportError as error:
            raise ImportError(
                'Wandb serializer requires the wandb python package. '
                'To get it, run "pip install wandb".'
            ) from error
        else:
            dedicated_run = set_wandb_run_state(
                dedicated_run,
                {'header': self.header, **self.check.metadata()},
                **kwargs
            )
            wandb.log(WandbSerializer(self).serialize())
            if dedicated_run:
                wandb.finish()

    def __repr__(self):
        """Return string representation."""
        return self.header + ': ' + str(self.exception)

    def _ipython_display_(self):
        """Display the check failure."""
        display_html(
            CheckFailureHtmlSerializer(self).serialize(),
            raw=True
        )

    def print_traceback(self):
        """Print the traceback of the failure."""
        print(''.join(traceback.format_exception(
            etype=type(self.exception),
            value=self.exception,
            tb=self.exception.__traceback__
        )))
