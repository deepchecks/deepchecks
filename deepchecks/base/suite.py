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
from collections import OrderedDict
from typing import Union, List, Optional, Tuple, Any, Container

from deepchecks.base.display_suite import display_suite_result, ProgressBar
from deepchecks.errors import DeepchecksValueError
from deepchecks.base import Dataset
from deepchecks.base.check import (CheckResult, TrainTestBaseCheck,
                                   SingleDatasetBaseCheck, ModelOnlyBaseCheck, CheckFailure, ModelComparisonBaseCheck)
from deepchecks.utils.ipython import is_ipython_display


__all__ = ['Suite', 'ModelComparisonSuite', 'SuiteResult']


class SuiteResult:
    """Contain the results of a suite run."""

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
        display_suite_result(self.name, self.results)

    def show(self):
        """Display suite result."""
        if is_ipython_display():
            self._ipython_display_()
        else:
            print(self)


class BaseSuite:
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
        name: Name of the suite
    """

    @classmethod
    @abc.abstractmethod
    def supported_checks(cls) -> Tuple:
        pass

    checks: OrderedDict
    name: str
    _check_index: int

    def __init__(self, name: str, *checks):
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

    def add(self, check):
        """Add a check or a suite to current suite.

        Args:
            check (BaseCheck): A check or suite to add.
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

        Args:
            index (int): Index of check to remove.
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
            train_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            model: object = None,
    ) -> SuiteResult:
        """Run all checks.

        Args:
          train_dataset: Dataset object, representing data an estimator was fitted on
          test_dataset: Dataset object, representing data an estimator predicts on
          model: A scikit-learn-compatible fitted estimator instance

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if all(it is None for it in (train_dataset, test_dataset, model)):
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')

        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run(train_dataset=train_dataset, test_dataset=test_dataset,
                                                 model=model)
                        results.append(check_result)
                elif isinstance(check, SingleDatasetBaseCheck):
                    if train_dataset is not None:
                        check_result = check.run(dataset=train_dataset, model=model)
                        check_result.header = f'{check_result.get_header()} - Train Dataset'
                        results.append(check_result)
                    if test_dataset is not None:
                        check_result = check.run(dataset=test_dataset, model=model)
                        check_result.header = f'{check_result.get_header()} - Test Dataset'
                        results.append(check_result)
                elif isinstance(check, ModelOnlyBaseCheck):
                    if model is not None:
                        check_result = check.run(model=model)
                        results.append(check_result)
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check.__class__, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)


class ModelComparisonSuite(BaseSuite):
    """Suite to run checks of types: CompareModelsBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return tuple([ModelComparisonBaseCheck])

    def run(self,
            train_datasets: Union[Dataset, Container[Dataset]],
            test_datasets: Union[Dataset, Container[Dataset]],
            models: Container[Any]
            ) -> SuiteResult:
        """Run all checks.

        Args:
          train_datasets: 1 or more dataset object, representing data an estimator was fitted on
          test_datasets: 1 or more dataset object, representing data an estimator was fitted on
          models: 2 or more scikit-learn-compatible fitted estimator instance

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if any(it is None for it in (train_datasets, test_datasets, models)):
            raise DeepchecksValueError('All parameters must be passed to the suite!')

        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                check_result = check.run(train_datasets, test_datasets, models)
                results.append(check_result)
            except Exception as exp:
                results.append(CheckFailure(check.__class__, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)
