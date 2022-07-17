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
import importlib
import io
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Set, Tuple, Type, Union, cast

import jsonpickle
from ipywidgets import Widget
from typing_extensions import TypedDict

from deepchecks.core import check_result as check_types
from deepchecks.core.checks import BaseCheck, CheckConfig
from deepchecks.core.display import DisplayableResult, save_as_html
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.serialization.abc import HTMLFormatter
from deepchecks.core.serialization.suite_result.html import SuiteResultSerializer as SuiteResultHtmlSerializer
from deepchecks.core.serialization.suite_result.ipython import SuiteResultSerializer as SuiteResultIPythonSerializer
from deepchecks.core.serialization.suite_result.json import SuiteResultSerializer as SuiteResultJsonSerializer
from deepchecks.core.serialization.suite_result.widget import SuiteResultSerializer as SuiteResultWidgetSerializer
from deepchecks.utils.strings import get_random_string, widget_to_html_string
from deepchecks.utils.wandb_utils import wandb_run

__all__ = ['BaseSuite', 'SuiteResult']


class SuiteConfig(TypedDict):
    name: str
    module_name: str
    checks: List[CheckConfig]


class SuiteResult(DisplayableResult):
    """Contain the results of a suite run.

    Parameters
    ----------
    name: str
    results: List[BaseCheckResult]
    extra_info: Optional[List[str]]
    """

    name: str
    extra_info: List[str]
    results: List['check_types.BaseCheckResult']

    def __init__(
        self,
        name: str,
        results: List['check_types.BaseCheckResult'],
        extra_info: Optional[List[str]] = None,
    ):
        """Initialize suite result."""
        self.name = name
        self.results = sort_check_results(results)
        self.extra_info = extra_info or []

        # NOTE:
        # we collect results indexes in order to facilitate results
        # filtering and selection via the `select_results` method
        #
        # Examples:
        # >>
        # >> sr.select_result(sr.results_with_conditions | sr.results_with_display)
        # >> sr.select_results(sr.results_without_conditions & sr.results_with_display)

        self.results_with_conditions: Set[int] = set()
        self.results_without_conditions: Set[int] = set()
        self.results_with_display: Set[int] = set()
        self.results_without_display: Set[int] = set()
        self.failures: Set[int] = set()

        for index, result in enumerate(self.results):
            if isinstance(result, check_types.CheckFailure):
                self.failures.add(index)
            elif isinstance(result, check_types.CheckResult):
                has_conditions = result.have_conditions()
                has_display = result.have_display()
                if has_conditions:
                    self.results_with_conditions.add(index)
                else:
                    self.results_without_conditions.add(index)
                if has_display:
                    self.results_with_display.add(index)
                else:
                    self.results_without_display.add(index)
            else:
                raise TypeError(f'Unknown type of result - {type(result).__name__}')

    def select_results(self, idx: Set[int]) -> List[Union[
        'check_types.CheckResult',
        'check_types.CheckFailure'
    ]]:
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
        requirejs: bool = False,
    ) -> str:
        """Return html representation of check result."""
        return widget_to_html_string(
            self.to_widget(unique_id=unique_id or get_random_string(n=25)),
            title=self.name,
            requirejs=requirejs
        )

    def _repr_json_(self):
        return SuiteResultJsonSerializer(self).serialize()

    def _repr_mimebundle_(self, **kwargs):
        return {
            'text/html': self._repr_html_(),
            'application/json': self._repr_json_()
        }

    @property
    def widget_serializer(self) -> SuiteResultWidgetSerializer:
        """Return WidgetSerializer instance."""
        return SuiteResultWidgetSerializer(self)

    @property
    def ipython_serializer(self) -> SuiteResultIPythonSerializer:
        """Return IPythonSerializer instance."""
        return SuiteResultIPythonSerializer(self)

    @property
    def html_serializer(self) -> SuiteResultHtmlSerializer:
        """Return HtmlSerializer instance."""
        return SuiteResultHtmlSerializer(self)

    def show(
        self,
        as_widget: bool = True,
        unique_id: Optional[str] = None,
        **kwargs
    ) -> Optional[HTMLFormatter]:
        """Display result.

        Parameters
        ----------
        as_widget : bool
            whether to display result with help of ipywidgets or not
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwrgs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method

        Returns
        -------
        Optional[HTMLFormatter] :
            when used by sphinx-gallery
        """
        return super().show(
            as_widget,
            unique_id or get_random_string(n=25),
            **kwargs
        )

    def show_not_interactive(
        self,
        unique_id: Optional[str] = None,
        **kwargs
    ):
        """Display the not interactive version of result output.

        In this case, ipywidgets will not be used and plotly
        figures will be transformed into png images.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output
        **kwrgs :
            other key-value arguments will be passed to the `Serializer.serialize`
            method
        """
        return super().show_not_interactive(
            unique_id or get_random_string(n=25),
            **kwargs
        )

    def save_as_html(
        self,
        file: Union[str, io.TextIOWrapper, None] = None,
        as_widget: bool = True,
        requirejs: bool = True,
        unique_id: Optional[str] = None,
        connected: bool = False,
        **kwargs
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
        unique_id : Optional[str], default None
            unique identifier of the result output
        connected: bool , default False
            indicates whether internet connection is available or not,
            if 'True' then CDN urls will be used to load javascript otherwise
            javascript libraries will be injected directly into HTML output.
            Set to 'False' to make results viewing possible when the internet
            connection is not available.

        Returns
        -------
        Optional[str] :
            name of newly create file
        """
        return save_as_html(
            file=file,
            serializer=self.widget_serializer if as_widget else self.html_serializer,
            connected=connected,
            # next kwargs will be passed to the serializer.serialize method
            requirejs=requirejs,
            output_id=unique_id or get_random_string(n=25),
        )

    def to_widget(
        self,
        unique_id: Optional[str] = None,
        **kwargs
    ) -> Widget:
        """Return SuiteResult as a ipywidgets.Widget instance.

        Parameters
        ----------
        unique_id : Optional[str], default None
            unique identifier of the result output

        Returns
        -------
        Widget
        """
        output_id = unique_id or get_random_string(n=25)
        return SuiteResultWidgetSerializer(self).serialize(output_id=output_id)

    def to_json(self, with_display: bool = True, **kwargs):
        """Return check result as json.

        Parameters
        ----------
        with_display : bool, default True
            whether to include serialized `SuiteResult.display` items into
            the output or not

        Returns
        -------
        str
        """
        return jsonpickle.dumps(
            SuiteResultJsonSerializer(self).serialize(with_display=with_display),
            unpicklable=False
        )

    def to_wandb(
        self,
        dedicated_run: Optional[bool] = None,
        **kwargs
    ):
        """Send suite result to wandb.

        Parameters
        ----------
        dedicated_run : bool
            whether to create a separate wandb run or not
            (deprecated parameter, does not have any effect anymore)
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
        from deepchecks.core.serialization.suite_result.wandb import SuiteResultSerializer as WandbSerializer

        if dedicated_run is not None:
            warnings.warn(
                '"dedicated_run" parameter is deprecated and does not have effect anymore. '
                'It will be remove in next versions.'
            )

        wandb_kwargs = {'config': {'name': self.name}}
        wandb_kwargs.update(**kwargs)

        with wandb_run(**wandb_kwargs) as run:
            run.log(WandbSerializer(self).serialize())

    def get_not_ran_checks(self) -> List['check_types.CheckFailure']:
        """Get all the check results which did not run (unable to run due to missing parameters, exception, etc).

        Returns
        -------
        List[CheckFailure]
            All the check failures in the suite.
        """
        return cast(List[check_types.CheckFailure], self.select_results(self.failures))

    def get_not_passed_checks(self, fail_if_warning=True) -> List['check_types.CheckResult']:
        """Get all the check results that have not passing condition.

        Parameters
        ----------
        fail_if_warning: bool, Default: True
            Whether conditions should fail on status of warning

        Returns
        -------
        List[CheckResult]
            All the check results in the suite that have failing conditions.
        """
        results = cast(
            List[check_types.CheckResult],
            self.select_results(self.results_with_conditions)
        )
        return [
            r for r in results
            if not r.passed_conditions(fail_if_warning)
        ]

    def get_passed_checks(self, fail_if_warning=True) -> List['check_types.CheckResult']:
        """Get all the check results that have passing condition.

        Parameters
        ----------
        fail_if_warning: bool, Default: True
            Whether conditions should fail on status of warning

        Returns
        -------
        List[CheckResult]
            All the check results in the suite that have failing conditions.
        """
        results = cast(
            List[check_types.CheckResult],
            self.select_results(self.results_with_conditions)
        )
        return [
            r for r in results
            if r.passed_conditions(fail_if_warning)
        ]

    def passed(self, fail_if_warning: bool = True, fail_if_check_not_run: bool = False) -> bool:
        """Return whether this suite result has passed. Pass value is derived from condition results of all individual\
         checks, and may consider checks that didn't run.

        Parameters
        ----------
        fail_if_warning: bool, Default: True
            Whether conditions should fail on status of warning
        fail_if_check_not_run: bool, Default: False
            Whether checks that didn't run (missing parameters, exception, etc) should fail the suite result.

        Returns
        -------
        bool
        """
        not_run_pass = len(self.get_not_ran_checks()) == 0 if fail_if_check_not_run else True
        conditions_pass = len(self.get_not_passed_checks(fail_if_warning)) == 0
        return conditions_pass and not_run_pass

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
            results.append(check_types.BaseCheckResult.from_json(res))
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

    def config(self) -> SuiteConfig:
        """Return suite configuration (checks' conditions' configuration not yet supported).

        Returns
        -------
        SuiteConfig
            includes the suite name, and list of check configs.
        """
        meta_data = SuiteConfig(name=self.name, checks=[], module_name=self.__module__)
        for check in self.checks.values():
            meta_data['checks'].append(check.config())
        return meta_data

    @staticmethod
    def from_config(conf: SuiteConfig) -> 'BaseSuite':
        """Return suite object from a CheckConfig object.

        Parameters
        ----------
        conf : SuiteConfig
            the SuiteConfig object

        Returns
        -------
        BaseSuite
            the suite class object from given config
        """
        checks = []
        for check_conf in conf['checks']:
            checks.append(BaseCheck.from_config(check_conf))

        module = importlib.import_module(conf['module_name'])
        suite_cls: Type[BaseSuite] = getattr(module, 'Suite')
        return suite_cls(conf['name'], *checks)


def sort_check_results(
    check_results: Sequence['check_types.BaseCheckResult']
) -> List['check_types.BaseCheckResult']:
    """Sort sequence of 'CheckResult' instances.

    Returns
    -------
    List[check_types.CheckResult]
    """
    order = []
    check_results_index = {}

    for index, it in enumerate(check_results):
        check_results_index[index] = it

        if isinstance(it, check_types.CheckResult):
            order.append((it.priority, index))
        elif isinstance(it, check_types.CheckFailure):
            order.append((998, index))
        else:
            order.append((999, index))

    order = sorted(order)

    return [
        check_results_index[index]
        for _, index in order
    ]
