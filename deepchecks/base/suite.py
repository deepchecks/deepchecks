# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the Suite object, used for running a set of checks together."""
# pylint: disable=broad-except
import abc
import io
import warnings
from collections import OrderedDict
from typing import Union, List, Optional, Tuple, Any, Container, Mapping, Callable

import pandas as pd
from IPython.core.display import display_html
from IPython.core.getipython import get_ipython
import jsonpickle

from deepchecks.base.check_context import CheckRunContext
from deepchecks.base.display_suite import display_suite_result, ProgressBar
from deepchecks.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.base.dataset import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck, SingleDatasetBaseCheck, ModelOnlyBaseCheck, \
                                  CheckFailure, ModelComparisonBaseCheck, ModelComparisonContext, BaseCheck
from deepchecks.utils.ipython import is_notebook
from deepchecks.utils.typing import BasicModel


__all__ = ['BaseSuite', 'Suite', 'ModelComparisonSuite', 'SuiteResult']


class SuiteResult:
    """Contain the results of a suite run.

    Parameters
    ----------
    name: str
    results: List[Union[CheckResult, CheckFailure]]
    """

    name: str
    results: List[Union[CheckResult, CheckFailure]]

    def __init__(self, name: str, results):
        """Initialize suite result."""
        self.name = name
        self.results = results

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.name

    def _ipython_display_(self):
        # google colab has no support for widgets but good support for viewing html pages in the output
        if 'google.colab' in str(get_ipython()):
            html_out = io.StringIO()
            display_suite_result(self.name, self.results, html_out=html_out)
            display_html(html_out.getvalue(), raw=True)
        else:
            display_suite_result(self.name, self.results)

    def show(self):
        """Display suite result."""
        if is_notebook():
            self._ipython_display_()
        else:
            warnings.warn('You are running in a non-interactive python shell. in order to show result you have to use '
                          'an IPython shell (etc Jupyter)')

    def save_as_html(self, file=None):
        """Save output as html file.

        Parameters
        ----------
        file : filename or file-like object
            The file to write the HTML output to. If None writes to output.html
        """
        if file is None:
            file = 'output.html'
        display_suite_result(self.name, self.results, html_out=file)

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


class Suite(BaseSuite):
    """Suite to run checks of types: TrainTestBaseCheck, SingleDatasetBaseCheck, ModelOnlyBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestBaseCheck, SingleDatasetBaseCheck, ModelOnlyBaseCheck

    def run(
            self,
            train_dataset: Optional[Union[Dataset, pd.DataFrame]] = None,
            test_dataset: Optional[Union[Dataset, pd.DataFrame]] = None,
            model: BasicModel = None,
            features_importance: pd.Series = None,
            feature_importance_force_permutation: bool = False,
            feature_importance_timeout: int = None,
            scorers: Mapping[str, Union[str, Callable]] = None,
            scorers_per_class: Mapping[str, Union[str, Callable]] = None
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator was fitted on
        test_dataset : Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator predicts on
        model : BasicModel , default None
            A scikit-learn-compatible fitted estimator instance
        features_importance : pd.Series , default None
            pass manual features importance
        feature_importance_force_permutation : bool , default None
            force calculation of permutation features importance
        feature_importance_timeout : int , default None
            timeout in second for the permutation features importance calculation
        scorers : Mapping[str, Union[str, Callable]] , default None
            dict of scorers names to scorer sklearn_name/function
        scorers_per_class : Mapping[str, Union[str, Callable]], default None
            dict of scorers for classification without averaging of the classes
            See <a href=
            "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
            scikit-learn docs</a>
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = CheckRunContext(train_dataset, test_dataset, model,
                                  features_importance=features_importance,
                                  feature_importance_force_permutation=feature_importance_force_permutation,
                                  feature_importance_timeout=feature_importance_timeout,
                                  scorers=scorers,
                                  scorers_per_class=scorers_per_class)
        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if not supplied with both train and test datasets'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, SingleDatasetBaseCheck):
                    if train_dataset is not None:
                        # In case of train & test, doesn't want to skip test if train fails. so have to explicitly
                        # wrap it in try/except
                        try:
                            check_result = check.run_logic(context)
                            # In case of single dataset not need to edit the header
                            if test_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Train Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Train Dataset')
                        results.append(check_result)
                    if test_dataset is not None:
                        try:
                            check_result = check.run_logic(context, dataset_type='test')
                            # In case of single dataset not need to edit the header
                            if train_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Test Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Test Dataset')
                        results.append(check_result)
                    if train_dataset is None and test_dataset is None:
                        msg = 'Check is irrelevant if dataset is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, ModelOnlyBaseCheck):
                    if model is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if model is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))


class ModelComparisonSuite(BaseSuite):
    """Suite to run checks of types: CompareModelsBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return tuple([ModelComparisonBaseCheck])

    def run(self,
            train_datasets: Union[Dataset, Container[Dataset]],
            test_datasets: Union[Dataset, Container[Dataset]],
            models: Union[Container[Any], Mapping[str, Any]]
            ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_datasets : Union[Dataset, Container[Dataset]]
            representing data an estimator was fitted on
        test_datasets: Union[Dataset, Container[Dataset]]
            representing data an estimator was fitted on
        models : Union[Container[Any], Mapping[str, Any]]
            2 or more scikit-learn-compatible fitted estimator instance
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        Raises
        ------
        ValueError
            if check_datasets_policy is not of allowed types
        """
        context = ModelComparisonContext(train_datasets, test_datasets, models)

        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                check_result = check.run_logic(context)
                results.append(check_result)
            except Exception as exp:
                results.append(CheckFailure(check, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)
