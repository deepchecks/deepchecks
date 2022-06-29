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
from deepchecks.core.errors import DeepchecksBaseError
from deepchecks.tabular import Context, SingleDatasetCheck, checks
from deepchecks.tabular.datasets.classification import lending_club
from deepchecks.tabular.datasets.regression import avocado


def run_check_fn(check_class) -> Callable:
    def run(self, cache, dataset_name):
        context = cache[dataset_name]
        check = check_class()
        try:
            if isinstance(check, SingleDatasetCheck):
                check.run_logic(context, DatasetKind.TRAIN)
            else:
                check.run_logic(context)
        except DeepchecksBaseError:
            pass
    return run


def setup_lending_club() -> Context:
    train, test = lending_club.load_data()
    model = lending_club.load_fitted_model()
    context = Context(train, test, model)
    context.feature_importance  # calculating here to avoid first check being slower
    return context


def setup_avocado() -> Context:
    train, test = avocado.load_data()
    model = avocado.load_fitted_model()
    context = Context(train, test, model)
    context.feature_importance  # calculating here to avoid first check being slower
    return context


class BenchmarkTabular:
    params = ['lending_club', 'avocado']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        cache['lending_club'] = setup_lending_club()
        cache['avocado'] = setup_avocado()
        return cache


for name, check_class in inspect.getmembers(checks):
    if inspect.isclass(check_class):
        run_fn = run_check_fn(check_class)
        setattr(BenchmarkTabular, f'time_{name}', run_fn)
        setattr(BenchmarkTabular, f'peakmem_{name}', run_fn)
