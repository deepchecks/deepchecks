# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
from typing import Dict

import numpy as np
from sklearn.metrics import make_scorer, jaccard_score

from hamcrest import assert_that, close_to, has_items, has_length
from ignite.engine import Engine
from ignite.metrics import Metric
from numpy import nanmean

from deepchecks.vision import VisionData
from deepchecks.vision.metrics_utils import get_scorers_dict
from deepchecks.vision.metrics_utils.detection_precision_recall import ObjectDetectionAveragePrecision
from deepchecks.vision.metrics_utils.semantic_segmentation_metrics import MeanDice, MeanIoU, per_sample_dice
from deepchecks.vision.vision_data.utils import sequence_to_numpy


def calculate_metrics(metrics: Dict[str, Metric], dataset: VisionData) -> Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Returns
    -------
    t.Dict[str, float]
        Dictionary of metrics with the metric name as key and the metric value as value
    """

    def process_function(_, batch):
        return sequence_to_numpy(batch['predictions']), sequence_to_numpy(batch['labels'])

    engine = Engine(process_function)
    for name, metric in metrics.items():
        metric.reset()
        metric.attach(engine, name)

    state = engine.run(dataset.batch_loader)
    return state.metrics


def test_default_ap_ignite_complient(coco_visiondata_test):
    res = calculate_metrics({'AveragePrecision': ObjectDetectionAveragePrecision()},
                            coco_visiondata_test)
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(80))


def test_ar_ignite_complient(coco_visiondata_test):
    res = calculate_metrics({'AverageRecall': ObjectDetectionAveragePrecision(return_option='ar')},
                            coco_visiondata_test)

    assert_that(res.keys(), has_length(1))
    assert_that(res['AverageRecall'], has_length(80))


def test_equal_pycocotools(coco_visiondata_test):
    metric = ObjectDetectionAveragePrecision(return_option=None)
    for batch in coco_visiondata_test:
        metric.update((sequence_to_numpy(batch['predictions']), sequence_to_numpy(batch['labels'])))
    res = metric.compute()[0]

    assert_that(metric.get_classes_scores_at(res['precision'], area='all', max_dets=100), close_to(0.409, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], iou=0.5, area='all', max_dets=100),
                close_to(0.566, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], iou=0.75, area='all', max_dets=100),
                close_to(0.425, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='small', max_dets=100), close_to(0.212, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='medium', max_dets=100), close_to(0.383, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='large', max_dets=100), close_to(0.541, 0.001))

    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=1), close_to(0.330, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=10), close_to(0.423, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=100), close_to(0.429, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='small', max_dets=100), close_to(0.220, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='medium', max_dets=100), close_to(0.423, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='large', max_dets=100), close_to(0.549, 0.001))

    # unrelated to pycoco but needed to check another param
    assert_that(metric.get_classes_scores_at(res['recall'], area='large', max_dets=100, get_mean_val=False,
                zeroed_negative=False), has_items([-1]))
    assert_that(metric.get_classes_scores_at(res['recall'], get_mean_val=False, zeroed_negative=False), has_items([-1]))


def test_average_precision_recall(coco_visiondata_test):
    res = calculate_metrics({'ap': ObjectDetectionAveragePrecision(),
                             'ap_macro': ObjectDetectionAveragePrecision(average='macro'),
                             'ap_weighted': ObjectDetectionAveragePrecision(average='weighted')},
                            coco_visiondata_test)
    # classes mean and macro are not equal due to zeroed negative
    assert_that(nanmean(res['ap']), close_to(0.396, 0.001))
    assert_that(res['ap_macro'], close_to(0.409, 0.001))
    assert_that(res['ap_weighted'], close_to(0.441, 0.001))


def test_average_precision_thresholds(coco_visiondata_test):
    res = calculate_metrics({'ap': ObjectDetectionAveragePrecision(iou_range=(0.4, 0.8, 5), average='macro')},
                            coco_visiondata_test)
    assert_that(res['ap'], close_to(0.514, 0.001))


def test_segmentation_metrics(segmentation_coco_visiondata_train):
    dice_per_class = MeanDice()
    dice_micro = MeanDice(average='micro')
    dice_macro = MeanDice(average='macro')
    iou_per_class = MeanIoU()
    iou_micro = MeanIoU(average='micro')
    iou_macro = MeanIoU(average='macro')

    for batch in segmentation_coco_visiondata_train:
        label = sequence_to_numpy(batch['labels'])
        prediction = sequence_to_numpy(batch['predictions'])
        dice_per_class.update((prediction, label))
        dice_micro.update((prediction, label))
        dice_macro.update((prediction, label))
        iou_per_class.update((prediction, label))
        iou_micro.update((prediction, label))
        iou_macro.update((prediction, label))
    assert_that(dice_per_class.compute()[0], close_to(0.973, 0.001))
    assert_that(dice_per_class.compute(), has_length(17))
    assert_that(dice_micro.compute(), close_to(0.951, 0.001))
    assert_that(dice_macro.compute(), close_to(0.649, 0.006))
    assert_that(iou_per_class.compute()[0], close_to(0.948, 0.001))


def test_per_sample_dice(segmentation_coco_visiondata_train):
    batch = next(iter(segmentation_coco_visiondata_train))
    res = per_sample_dice(batch['predictions'], batch['labels'])
    assert_that(sum(res), close_to(9.513, 0.001))


def test_string_metric_classification(mnist_visiondata_test):
    metric_dict = get_scorers_dict(mnist_visiondata_test, {'acc': 'accuracy'})
    res = calculate_metrics(metric_dict, mnist_visiondata_test)
    assert_that(res['acc'], close_to(0.985, 0.001))


def test_scorer_metric_classification(mnist_visiondata_test):
    scorer = make_scorer(jaccard_score, average=None, zero_division=0)
    metric_dict = get_scorers_dict(mnist_visiondata_test, {'kappa': scorer})
    res = calculate_metrics(metric_dict, mnist_visiondata_test)
    assert_that(np.mean(list(res['kappa'].values())), close_to(0.976, 0.001))
