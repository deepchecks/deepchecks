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
import logging
import sys
from typing import Callable

import torch

from deepchecks.core.errors import DeepchecksBaseError
from deepchecks.vision import Context, SingleDatasetCheck, TrainTestCheck, ModelOnlyCheck, checks
from deepchecks.vision.datasets.classification import mnist
from deepchecks.vision.datasets.detection import coco


# need this code so pickle won't fail on the coco model (asv using pickle on the cache for some reason)
logger = logging.getLogger('yolov5')
logger.disabled = True
_ = torch.hub.load('ultralytics/yolov5:v6.1', 'yolov5s',
                        pretrained=True,
                        verbose=False,
                        device='cpu')
sys.path.append('ultralytics_yolov5_v6.1/models')


def run_check_fn(check_class) -> Callable:
    def run(self, cache, dataset_name):
        context: Context = cache[dataset_name]
        check = check_class()
        try:
            if isinstance(check, SingleDatasetCheck):
                check.run(context.train, context.model)
            elif isinstance(check, TrainTestCheck):
                check.run(context.train, context.test, context.model)
            elif isinstance(check, ModelOnlyCheck):
                check.run(context.model)
        except DeepchecksBaseError:
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


class BenchmarkVision:
    params = ['coco', 'mnist']
    param_names = ['dataset_name']

    def setup_cache(self):
        cache = {}
        cache['mnist'] = setup_mnist()
        cache['coco'] = setup_coco()
        return cache


for name, check_class in inspect.getmembers(checks):
    if inspect.isclass(check_class):
        run_fn = run_check_fn(check_class)
        setattr(BenchmarkVision, f'time_{name}', run_fn)
        setattr(BenchmarkVision, f'peakmem_{name}', run_fn)
