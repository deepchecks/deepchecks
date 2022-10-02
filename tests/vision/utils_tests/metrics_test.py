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
from hamcrest import assert_that, close_to, has_items, has_length

from deepchecks.vision import VisionData
from deepchecks.vision.metrics_utils.detection_precision_recall import ObjectDetectionAveragePrecision
from deepchecks.vision.metrics_utils.scorers import calculate_metrics
from deepchecks.vision.metrics_utils.semantic_segmentation_metrics import MeanDice, MeanIoU
from numpy import nanmean


def test_default_ap_ignite_complient(coco_test_visiondata: VisionData, mock_trained_yolov5_object_detection, device):
    res = calculate_metrics({'AveragePrecision': ObjectDetectionAveragePrecision()},
                            coco_test_visiondata, mock_trained_yolov5_object_detection,
                            device=device)
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(80))


def test_ar_ignite_complient(coco_test_visiondata: VisionData, mock_trained_yolov5_object_detection, device):
    res = calculate_metrics({'AverageRecall': ObjectDetectionAveragePrecision(return_option='ar')},
                            coco_test_visiondata, mock_trained_yolov5_object_detection,
                            device=device)

    assert_that(res.keys(), has_length(1))
    assert_that(res['AverageRecall'], has_length(80))


def test_equal_pycocotools(coco_test_visiondata: VisionData, mock_trained_yolov5_object_detection, device):
    metric = ObjectDetectionAveragePrecision(return_option=None)
    for batch in coco_test_visiondata:
        label = coco_test_visiondata.batch_to_labels(batch)
        prediction = coco_test_visiondata.infer_on_batch(batch, mock_trained_yolov5_object_detection, device)
        metric.update((prediction, label))
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


def test_average_precision_recall(coco_test_visiondata: VisionData, mock_trained_yolov5_object_detection, device):
    res = calculate_metrics({'ap': ObjectDetectionAveragePrecision(),
                             'ap_macro': ObjectDetectionAveragePrecision(average='macro'),
                             'ap_weighted': ObjectDetectionAveragePrecision(average='weighted')},
                            coco_test_visiondata, mock_trained_yolov5_object_detection,
                            device=device)
    # classes mean and macro are not equal due to zeroed negative
    assert_that(nanmean(res['ap']), close_to(0.396, 0.001))
    assert_that(res['ap_macro'], close_to(0.409, 0.001))
    assert_that(res['ap_weighted'], close_to(0.441, 0.001))


def test_segmentation_metrics(segmentation_coco_train_visiondata, trained_segmentation_deeplabv3_mobilenet_model,
                              device):
    dice_per_class = MeanDice()
    dice_micro = MeanDice(average='micro')
    dice_macro = MeanDice(average='macro')
    iou_per_class = MeanIoU()
    iou_micro = MeanIoU(average='micro')
    iou_macro = MeanIoU(average='macro')

    for batch in segmentation_coco_train_visiondata:
        label = segmentation_coco_train_visiondata.batch_to_labels(batch)
        prediction = segmentation_coco_train_visiondata.infer_on_batch(
            batch, trained_segmentation_deeplabv3_mobilenet_model, device)
        dice_per_class.update((prediction, label))
        dice_micro.update((prediction, label))
        dice_macro.update((prediction, label))
        iou_per_class.update((prediction, label))
        iou_micro.update((prediction, label))
        iou_macro.update((prediction, label))
    assert_that(dice_per_class.compute()[0], close_to(0.973, 0.001))
    assert_that(dice_per_class.compute(), has_length(17))
    assert_that(dice_micro.compute(), close_to(0.951, 0.001))
    assert_that(dice_macro.compute(), close_to(0.649, 0.001))
    assert_that(iou_per_class.compute()[0], close_to(0.948, 0.001))
