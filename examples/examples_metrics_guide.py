# Default metrics example
from deepchecks.vision.checks import ClassPerformance
from deepchecks.vision.datasets.classification import mnist
mnist_model = mnist.load_model()
train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')
check = ClassPerformance()
result = check.run(train_ds, test_ds, mnist_model)

# Passing strings example
from deepchecks.tabular.checks import TrainTestPerformance
from deepchecks.tabular.datasets.classification import adult
train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
model = adult.load_fitted_model()

scorer = ['precision_per_class', 'recall_per_class', 'fnr_macro']
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

# Vision custom metric example
from ignite.metrics import Precision
from deepchecks.vision.checks import SingleDatasetPerformance

precision = Precision(average=True)
double_precision = 2 * precision

check = SingleDatasetPerformance(scorers={'precision2': double_precision})
result = check.run(train_ds, mnist_model)

# Deepchecks metric example
from deepchecks.vision.metrics import MeanDice
from deepchecks.vision.datasets.segmentation.segmentation_coco import load_dataset, load_model

coco_dataset = load_dataset()
coco_model = load_model()
metric = {'mean_dice': MeanDice()}

check = SingleDatasetPerformance(scorers=metric)
result = check.run(coco_dataset, coco_model)
