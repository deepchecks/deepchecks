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
from collections import OrderedDict
from typing import Union, List, Optional

from deepchecks.base.display_suite import display_suite_result, ProgressBar
from deepchecks.errors import DeepchecksValueError
from deepchecks.base import Dataset
from deepchecks.base.check import (
    BaseCheck, CheckResult, TrainTestBaseCheck,
    SingleDatasetBaseCheck, ModelOnlyBaseCheck, CheckFailure
)


__all__ = ['Suite', 'SuiteResult']


class SuiteResult:
    """Contain the results of a suite run."""

    name: str
    results: List[Union[CheckResult, CheckFailure]]

    def __init__(self, name: str, results):
        """Initialize suite result."""
        self.name = name
        self.results = results

    def _ipython_display_(self):
        display_suite_result(self.name, self.results)


class Suite(BaseCheck):
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
    """

    checks: OrderedDict
    name: str
    _check_index: int

    def __init__(self, name: str, *checks):
        """Get 'Check's and 'Suite's to run in given order."""
        super().__init__()
        self.name = name
        self.checks = OrderedDict()
        self._check_index = 0
        for check in checks:
            self.add(check)

    def run(
        self,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        model: object = None,
        check_datasets_policy: str = 'test'
    ) -> SuiteResult:
        """Run all checks.

        Args:
          model: A scikit-learn-compatible fitted estimator instance
          train_dataset: Dataset object, representing data an estimator was fitted on
          test_dataset: Dataset object, representing data an estimator predicts on
          check_datasets_policy: str, one of either ['both', 'train', 'test'].
                                 Determines the policy by which single dataset checks are run when two datasets are
                                 given, one for train and the other for test.

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if check_datasets_policy not in ['both', 'train', 'test']:
            raise ValueError('check_datasets_policy must be one of ["both", "train", "test"]')

        if all(it is None for it in (train_dataset, test_dataset, model)):
            raise ValueError('At least one dataset (or model) must be passed to the method!')

        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            check.set_conditions_display(False)
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run(train_dataset=train_dataset, test_dataset=test_dataset,
                                                 model=model)
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, SingleDatasetBaseCheck):
                    if check_datasets_policy in ['both', 'train'] and train_dataset is not None:
                        check_result = check.run(dataset=train_dataset, model=model)
                        check_result.header = f'{check_result.get_header()} - Train Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                    if check_datasets_policy in ['both', 'test'] and test_dataset is not None:
                        check_result = check.run(dataset=test_dataset, model=model)
                        check_result.header = f'{check_result.get_header()} - Test Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, ModelOnlyBaseCheck):
                    if model is not None:
                        check_result = check.run(model=model)
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check.__class__, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)

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
        if not isinstance(check, BaseCheck):
            raise DeepchecksValueError(
                f'Suite receives only `BaseCheck` objects but got: {check.__class__.__name__}'
            )
        if isinstance(check, Suite):
            if check is self:
                return self
            for c in check.checks.values():
                self.add(c)
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
