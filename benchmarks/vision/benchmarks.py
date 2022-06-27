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
from deepchecks.vision.vision_data import VisionData


# need this code so pickle won't fail on the coco model (asv using pickle on the cache for some reason)
logging.getLogger('yolov5').disabled = True
_ = torch.hub.load('ultralytics/yolov5:v6.1', 'yolov5s',
                        pretrained=True,
                        verbose=False,
                        device='cpu')
sys.path.append('ultralytics_yolov5_v6.1/models')

# logging.getLogger('deepchecks').setLevel(logging.ERROR)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_static_predictions(train: VisionData, test: VisionData, model):
    static_preds = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_pred = {}
            for i, batch in enumerate(vision_data):
                predictions = vision_data.infer_on_batch(batch, model, device)
                indexes = list(vision_data.data_loader.batch_sampler)[i]
                static_pred.update(dict(zip(indexes, predictions)))
        else:
            static_pred = None
        static_preds.append(static_pred)
    train_preds, tests_preds = static_preds
    return train_preds, tests_preds


def run_check_fn(check_class) -> Callable:
    def run(self, cache, dataset_name):
        train_ds, test_ds, model, train_pred, test_pred = cache[dataset_name]
        check = check_class()
        try:
            if isinstance(check, SingleDatasetCheck):
                check.run(train_ds, train_predictions=train_pred, device=device)
            elif isinstance(check, TrainTestCheck):
                check.run(train_ds, test_ds, train_predictions=train_pred,
                          test_predictions=test_pred, device=device)
            elif isinstance(check, ModelOnlyCheck):
                check.run(model, device=device)
        except DeepchecksBaseError:
            pass
    return run


def setup_mnist() -> Context:
    mnist_model = mnist.load_model()
    train_ds = mnist.load_dataset(train=True, object_type='VisionData')
    test_ds = mnist.load_dataset(train=False, object_type='VisionData')
    train_preds, tests_preds = create_static_predictions(train_ds, test_ds, mnist_model)
    return train_ds, test_ds, mnist_model, train_preds, tests_preds


def setup_coco() -> Context:
    coco_model = coco.load_model()
    train_ds = coco.load_dataset(train=True, object_type='VisionData')
    test_ds = coco.load_dataset(train=False, object_type='VisionData')
    train_preds, tests_preds = create_static_predictions(train_ds, test_ds, coco_model)
    return train_ds, test_ds, coco_model, train_preds, tests_preds


class BenchmarkVision:
    timeout = 120
    params = ['coco', 'mnist']
    param_names = ['dataset_name']
    repeat = (1, 3, 10)
    warmup_time = 0
    rounds = 1

    def setup_cache(self):
        cache = {}
        cache['mnist'] = setup_mnist()
        cache['coco'] = setup_coco()
        return cache


for name, check_class in inspect.getmembers(checks):
    if inspect.isclass(check_class):
        run_fn = run_check_fn(check_class)
        setattr(BenchmarkVision, f'time_{name}', run_fn)
        # setattr(BenchmarkVision, f'peakmem_{name}', run_fn)
