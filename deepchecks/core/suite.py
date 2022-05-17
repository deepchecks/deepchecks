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
# pylint: disable=unused-argument, import-outside-toplevel
"""Module containing the Suite object, used for running a set of checks together."""
import abc
import io
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union

import jsonpickle
from IPython.core.display import display, display_html
from ipywidgets import Widget

from deepchecks.core.check_result import (BaseCheckResult, CheckFailure,
                                          CheckResult)
from deepchecks.core.checks import BaseCheck
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.serialization.suite_result.html import \
    SuiteResultSerializer as SuiteResultHtmlSerializer
from deepchecks.core.serialization.suite_result.json import \
    SuiteResultSerializer as SuiteResultJsonSerializer
from deepchecks.core.serialization.suite_result.widget import \
    SuiteResultSerializer as SuiteResultWidgetSerializer
from deepchecks.utils.ipython import (is_colab_env, is_kaggle_env, is_notebook,
                                      is_widgets_enabled)
from deepchecks.utils.strings import (create_new_file_name, get_random_string,
                                      widget_to_html, widget_to_html_string)
from deepchecks.utils.wandb_utils import set_wandb_run_state

from .serialization.suite_result.ipython import \
    SuiteResultSerializer as SuiteResultIPythonSerializer

__all__ = ['BaseSuite', 'SuiteResult']


class SuiteResult:
    """Contain the results of a suite run.

    Parameters
    ----------
    name: str
    results: List[BaseCheckResult]
    """

    name: str
    extra_info: List[str]
    results: List[BaseCheckResult]

    def __init__(self, name: str, results, extra_info: Optional[List[str]] = None):
        """Initialize suite result."""
        self.name = name
        self.results = results
        self.extra_info = extra_info or []

        # TODO: add comment about code below

        self.results_with_conditions: Set[int] = set()
        self.results_without_conditions: Set[int] = set()
        self.results_with_display: Set[int] = set()
        self.results_without_display: Set[int] = set()
        self.failures: Set[int] = set()

        for index, result in enumerate(self.results):
            if isinstance(result, CheckResult):
                if result.have_conditions():
                    self.results_with_conditions.add(index)
                else:
                    self.results_without_conditions.add(index)
                if result.have_display():
                    self.results_with_display.add(index)
                else:
                    self.results_without_display.add(index)
            else:
                self.failures.add(index)

    def select_results(self, idx: Set[int]) -> List[BaseCheckResult]:
        """Select results by indexes."""
        output = []
        for index, result in enumerate(self.results):
            if index in idx:
                output.append(result)
        return output

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.name

    def _repr_html_(
        self,
        unique_id: Optional[str] = None,
        requirejs: bool = True,
    ) -> str:
        """Return html representation of check result."""
        output_id = (
            unique_id or get_random_string(n=25)
            if not is_colab_env()
            else None
        )
        return SuiteResultHtmlSerializer(self).serialize(
            include_plotlyjs=True,
            include_requirejs=requirejs,
            output_id=output_id
        )

    def _repr_json_(self):
        return SuiteResultJsonSerializer(self).serialize()

    def _repr_mimebundle_(self, **kwargs):
        return {
            'text/html': self._repr_html_(),
            'application/json': self._repr_json_()
        }

    def _ipython_display_(
        self,
        as_widget: bool = True,
        unique_id: Optional[str] = None
    ):
        output_id = (
            unique_id or get_random_string(n=25)
            if not is_colab_env()
            else None
        )
        if is_colab_env() and as_widget:
            display_html(
                widget_to_html_string(
                    self.to_widget(unique_id=output_id),
                    title=self.name,
                    requirejs=True
                ),
                raw=True
            )
        elif is_widgets_enabled() and as_widget and not is_kaggle_env():
            display_html(self.to_widget(unique_id=output_id))
        else:
            if as_widget:
                warnings.warn(
                    'Widgets are not enabled (or not supported) '
                    'and cannot be used.'
                )
            display(*SuiteResultIPythonSerializer(self).serialize(
                output_id=output_id,
            ))

    def show(
        self,
        as_widget: bool = True,
        unique_id: Optional[str] = None
    ):
        """Display suite result."""
        if is_notebook():
            self._ipython_display_(as_widget=as_widget, unique_id=unique_id)
        else:
            warnings.warn(
                'You are running in a non-interactive python shell. '
                'In order to show result you have to use '
                'an IPython shell (etc Jupyter)'
            )

    def save_as_html(
        self,
        file: Union[str, io.TextIOWrapper, None] = None,
        as_widget: bool = True,
        requirejs: bool = True,
        unique_id: Optional[str] = None
    ):
        """Save output as html file.

        Parameters
        ----------
        file : filename or file-like object
            The file to write the HTML output to. If None writes to output.html
        as_widget : bool, default True
            whether to use ipywidgets or not
        requirejs: bool , default: True
            whether to include requirejs library into output HTML or not

        Returns
        -------
        Optional[str] :
            name of newly create file
        """
        if file is None:
            file = 'output.html'
        if isinstance(file, str):
            file = create_new_file_name(file)

        if as_widget is True:
            widget_to_html(
                widget=self.to_widget(unique_id=unique_id or get_random_string(n=25)),
                html_out=file,
                title=self.name,
                requirejs=requirejs
            )
        else:
            html = SuiteResultHtmlSerializer(self).serialize(
                output_id=unique_id or get_random_string(n=25),
                full_html=True,
                include_requirejs=requirejs,
                include_plotlyjs=True
            )
            if isinstance(file, str):
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(html)
            elif isinstance(file, io.StringIO):
                file.write(html)
            else:
                TypeError(f'Unsupported type of "file" parameter - {type(file)}')

        if isinstance(file, str):
            return file

    def to_widget(
        self,
        unique_id : Optional[str] = None,
    ) -> Widget:
        """Return SuiteResult as a ipywidgets.Widget instance.

        Parameters
        ----------
        unique_id : str
            The unique id given by the suite that displays the check.

        Returns
        -------
        Widget
        """
        return SuiteResultWidgetSerializer(self).serialize(output_id=unique_id)

    def to_json(self, with_display: bool = True):
        """Return check result as json.

        Parameters
        ----------
        with_display : bool
            controls if to serialize display of checks or not

        Returns
        -------
        str
        """
        # TODO: not sure if the `with_display` parameter is needed
        # add deprecation warning if it is not needed
        return jsonpickle.dumps(
            SuiteResultJsonSerializer(self).serialize(),
            unpicklable=False
        )

    def to_wandb(
        self,
        dedicated_run: Optional[bool] = None,
        **kwargs
    ):
        """Export suite result to wandb.

        Parameters
        ----------
        dedicated_run : bool , default: None
            If to initiate and finish a new wandb run.
            If None it will be dedicated if wandb.run is None.
        kwargs: Keyword arguments to pass to wandb.init.
                Default project name is deepchecks.
                Default config is the suite name.
        """
        # NOTE:
        # Wandb is not a default dependency
        # user should install it manually therefore we are
        # doing import within method to prevent premature ImportError
        # TODO:
        # Previous implementation used ProgressBar to show serialization progress
        try:
            import wandb

            from deepchecks.core.serialization.suite_result.wandb import \
                SuiteResultSerializer as WandbSerializer
        except ImportError as error:
            raise ImportError(
                'Wandb serializer requires the wandb python package. '
                'To get it, run "pip install wandb".'
            ) from error
        else:
            dedicated_run = set_wandb_run_state(
                dedicated_run,
                {'name': self.name},
                **kwargs
            )
            wandb.log(WandbSerializer(self).serialize())
            if dedicated_run:  # TODO: create context manager for this
                wandb.finish()

    def get_failures(self) -> Dict[str, CheckFailure]:
        """Get all the failed checks.

        Returns
        -------
        Dict[str, CheckFailure]
            All the check failures in the suite.
        """
        failures = {}
        for res in self.results:
            if isinstance(res, CheckFailure):
                failures[res.header] = res
        return failures

    @classmethod
    def from_json(cls, json_res: str):
        """Convert a json object that was returned from SuiteResult.to_json.

        Parameters
        ----------
        json_data: Union[str, Dict]
            Json data

        Returns
        -------
        SuiteResult
            A suite result object.
        """
        json_dict = jsonpickle.loads(json_res)
        name = json_dict['name']
        results = []
        for res in json_dict['results']:
            results.append(BaseCheckResult.from_json(res))
        return SuiteResult(name, results)


class BaseSuite:
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Parameters
    ----------
    checks: OrderedDict
        A list of checks to run.
    name: str
        Name of the suite
    """

    @classmethod
    @abc.abstractmethod
    def supported_checks(cls) -> Tuple:
        """Return list of of supported check types."""
        pass

    checks: 'OrderedDict[int, BaseCheck]'
    name: str
    _check_index: int

    def __init__(self, name: str, *checks: Union[BaseCheck, 'BaseSuite']):
        self.name = name
        self.checks = OrderedDict()
        self._check_index = 0
        for check in checks:
            self.add(check)

    def __repr__(self, tabs=0):
        """Representation of suite as string."""
        tabs_str = '\t' * tabs
        checks_str = ''.join([f'\n{c.__repr__(tabs + 1, str(n) + ": ")}' for n, c in self.checks.items()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, index):
        """Access check inside the suite by name."""
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        return self.checks[index]

    def add(self, check: Union['BaseCheck', 'BaseSuite']):
        """Add a check or a suite to current suite.

        Parameters
        ----------
        check : BaseCheck
            A check or suite to add.
        """
        if isinstance(check, BaseSuite):
            if check is self:
                return self
            for c in check.checks.values():
                self.add(c)
        elif not isinstance(check, self.supported_checks()):
            raise DeepchecksValueError(
                f'Suite received unsupported object type: {check.__class__.__name__}'
            )
        else:
            self.checks[self._check_index] = check
            self._check_index += 1
        return self

    def remove(self, index: int):
        """Remove a check by given index.

        Parameters
        ----------
        index : int
            Index of check to remove.
        """
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        self.checks.pop(index)
        return self
