from sklearn.ensemble import AdaBoostClassifier

from deepchecks.tabular.datasets.classification import breast_cancer, iris, phishing, adult
from deepchecks.tabular.datasets.regression import avocado
import joblib


train, _ = breast_cancer.load_data()
model = AdaBoostClassifier(random_state=0)
model.fit(train.data[train.features], train.data[train.label_name])

# model1 = breast_cancer.load_fitted_model(pretrained=False)
joblib.dump(model, './breast_cancer.joblib')

