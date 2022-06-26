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
from deepchecks.vision import SingleDatasetCheck, Context, checks
from deepchecks.vision.datasets.classification import mnist
from deepchecks.vision.datasets.detection import coco

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

def setup_mnist() -> Context:
    mnist_model = mnist.load_model()
    train_ds = mnist.load_dataset(train=True, object_type='VisionData')
    test_ds = mnist.load_dataset(train=False, object_type='VisionData')
    return Context(train_ds, test_ds, mnist_model, n_samples=None)


def setup_coco() -> Context:
    coco_model = coco.load_model()
    train_ds = coco.load_dataset(train=True, object_type='VisionData')
    test_ds = coco.load_dataset(train=False, object_type='VisionData')
    return Context(train_ds, test_ds, coco_model, n_samples=None)


class BenchmarkVisionChecksTime:
    params = ['mnist', 'coco']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        cache['mnist'] = setup_mnist()
        cache['coco'] = setup_coco()
        return cache


class BenchmarkVisionChecksPeakMemory:
    params = ['mnist', 'coco']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        cache['mnist'] = setup_mnist()
        cache['coco'] = setup_coco()
        return cache


for name, check_class in inspect.getmembers(checks):
    if inspect.isclass(check_class):
        setattr(BenchmarkVisionChecksTime, f'time_{name}', run_check_fn(check_class))
        setattr(BenchmarkVisionChecksPeakMemory, f'peakmem_{name}', run_check_fn(check_class))
