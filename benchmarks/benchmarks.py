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
import inspect
from typing import Callable

from deepchecks.core import DatasetKind
from deepchecks.tabular import SingleDatasetCheck, Context, checks
from deepchecks.tabular.datasets.classification import iris


def run_check_fn(check_class) -> Callable:
    def run(self, cache, dataset_name):
        context = cache[dataset_name]
        check = check_class()
        try:
            if isinstance(check, SingleDatasetCheck):
                check.run_logic(context, DatasetKind.TRAIN)
            else:
                check.run_logic(context)
        except Exception as e:
            pass
    return run


class BenchmarkTabularChecksTime:
    params = ['iris']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        train, test = iris.load_data()
        model = iris.load_fitted_model()
        cache['iris'] = Context(train, test, model)
        return cache


class BenchmarkTabularChecksPeakMemory:
    params = ['iris']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        train, test = iris.load_data()
        model = iris.load_fitted_model()
        cache['iris'] = Context(train, test, model)
        return cache


for name, check_class in inspect.getmembers(checks):
    if inspect.isclass(check_class):
        setattr(BenchmarkTabularChecksTime, f'time_{name}', run_check_fn(check_class))
        setattr(BenchmarkTabularChecksPeakMemory, f'peakmem_{name}', run_check_fn(check_class))
