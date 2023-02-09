# Passing strings example
from deepchecks.tabular.checks import TrainTestPerformance
from deepchecks.tabular.datasets.classification import adult
train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
model = adult.load_fitted_model()

scorer = ['precision_per_class', 'recall_per_class', 'fnr']
check = TrainTestPerformance(scorers=scorer)
result = check.run(train_ds, test_ds, model)

# Tabular custom scorer example
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.suites import model_evaluation
from sklearn.metrics import cohen_kappa_score, fbeta_score, make_scorer

f1_scorer = make_scorer(fbeta_score, labels=[0, 1], average=None, beta=0.2)
ck_scorer = make_scorer(cohen_kappa_score)
custom_scorers = {'f1': f1_scorer, 'cohen': ck_scorer}

train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
model = adult.load_fitted_model()
suite = model_evaluation(scorers=custom_scorers)
result = suite.run(train_ds, test_ds, model)

# Default metrics example
from deepchecks.vision.checks import ClassPerformance
from deepchecks.vision.datasets.classification import mnist_torch as mnist
train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')
check = ClassPerformance()
result = check.run(train_ds, test_ds)

# Vision custom metric example
import numpy as np
import typing as t
from deepchecks.vision.checks import SingleDatasetPerformance
from deepchecks.vision.metrics_utils import CustomMetric

# For simplicity, we will implement the accuracy metric, although it is already implemented in deepchecks and
# can be passed as a string, and even otherwise we'd recommend using the sklearn API for custom classification metrics.

class CustomAccuracy(CustomMetric):

    def __init__(self):
        super().__init__()

    def reset(self):
        self._correct = 0
        self._total = 0
        super().reset()

    def update(self, output: t.Tuple[t.List[np.ndarray], t.List[int]]):
        y_pred, y = output
        y_pred = np.array(y_pred).argmax(axis=1)
        y = np.array(y)
        self._correct += (y_pred == y).sum()
        self._total += y_pred.shape[0]
        super().update(output)

    def compute(self):
        return self._correct / self._total

check = SingleDatasetPerformance(scorers={'accuracy': CustomAccuracy()})
result = check.run(train_ds)

# Deepchecks metric example
from deepchecks.vision.metrics import MeanDice
from deepchecks.vision.datasets.segmentation.segmentation_coco import load_dataset, load_model

coco_dataset = load_dataset()
metric = {'mean_dice': MeanDice(threshold=0.9)}

check = SingleDatasetPerformance(scorers=metric)
result = check.run(coco_dataset)
