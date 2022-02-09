from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.checks.performance.robustness_report import RobustnessReport
from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
from deepchecks.vision.checks.performance import PerformanceReport


test = coco.load_dataset(train=False, batch_size=1000, object_type='VisionData')
model = coco.load_model()


pred_formatter = DetectionPredictionFormatter(coco.yolo_prediction_formatter)
result = RobustnessReport(prediction_formatter=pred_formatter).run(test, model)


# yolo = coco.load_model(pretrained=True)
#
#
# train_ds = coco.load_dataset(train=True, object_type='VisionData')
# test_ds = coco.load_dataset(train=False, object_type='VisionData')
#
# check = PerformanceReport(prediction_formatter=DetectionPredictionFormatter(coco.yolo_prediction_formatter))
# check.run(train_ds, train_ds, yolo)
