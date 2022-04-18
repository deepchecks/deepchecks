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
"""Module containing the Suite object, used for running a set of checks together."""
import io
import abc
import warnings
from collections import OrderedDict
from typing import Any, Union, List, Tuple, Dict

from IPython.core.display import display_html
from IPython.core.getipython import get_ipython
import jsonpickle

from deepchecks.core.display_suite import ProgressBar, display_suite_result
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.check_result import CheckResult, CheckFailure
from deepchecks.core.checks import BaseCheck
from deepchecks.utils.ipython import is_notebook
from deepchecks.utils.wandb_utils import set_wandb_run_state

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

__all__ = ['BaseSuite', 'SuiteResult']


class SuiteResult:
    """Contain the results of a suite run.

    Parameters
    ----------
    name: str
    results: List[Union[CheckResult, CheckFailure]]
    """

    name: str
    extra_info: List[str]
    results: List[Union[CheckResult, CheckFailure]]

    def __init__(self, name: str, results, extra_info: List[str] = None):
        """Initialize suite result."""
        self.name = name
        self.results = results
        self.extra_info = extra_info or []

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.name

    def _ipython_display_(self):
        # google colab has no support for widgets but good support for viewing html pages in the output
        if 'google.colab' in str(get_ipython()):
            html_out = io.StringIO()
            display_suite_result(self.name, self.results, self.extra_info, html_out=html_out)
            display_html(html_out.getvalue(), raw=True)
        else:
            display_suite_result(self.name, self.results, self.extra_info)

    def show(self):
        """Display suite result."""
        if is_notebook():
            self._ipython_display_()
        else:
            warnings.warn('You are running in a non-interactive python shell. in order to show result you have to use '
                          'an IPython shell (etc Jupyter)')

    def _repr_html_(self):
        """Return html representation of check result."""
        html_out = io.StringIO()
        self.save_as_html(html_out, requirejs=False)
        html_page = html_out.getvalue()
        return html_page

    def save_as_html(self, file=None, requirejs: bool = True):
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
        display_suite_result(self.name, self.results, self.extra_info, html_out=file, requirejs=requirejs)

    def to_json(self, with_display: bool = True):
        """Return check result as json.

        Parameters
        ----------
        with_display : bool
            controls if to serialize display of checks or not

        Returns
        -------
        dict
            {'name': .., 'results': ..}
        """
        json_results = []
        for res in self.results:
            json_results.append(res.to_json(with_display=with_display))

        return jsonpickle.dumps({'name': self.name, 'results': json_results})

    def to_wandb(self, dedicated_run: bool = None, **kwargs: Any):
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
        dedicated_run = set_wandb_run_state(dedicated_run, {'name': self.name}, **kwargs)
        progress_bar = ProgressBar(self.name, len(self.results), 'Result')
        for res in self.results:
            res.to_wandb(False)
            progress_bar.inc_progress()
        progress_bar.close()
        if dedicated_run:
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

    checks: OrderedDict
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
