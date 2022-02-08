from deepchecks.vision.datasets.classification.mnist import load_dataset, load_model

mnist_dataloader_test = load_dataset(train=False, batch_size=1000, object_type='VisionData')
model = load_model()

from deepchecks.vision.checks.performance.robustness_report import RobustnessReport
from deepchecks.vision.utils.classification_formatters import ClassificationPredictionFormatter
import torch.nn as nn

RobustnessReport(prediction_formatter=ClassificationPredictionFormatter(nn.Softmax(dim=1))).run(mnist_dataloader_test, model)