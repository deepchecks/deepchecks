import numpy as np
from hamcrest import assert_that, instance_of

from deepchecks.tabular.datasets.classification import breast_cancer, iris, phishing, adult
from deepchecks.tabular.datasets.regression import avocado


def test_model_predict_on_dataset():
    datasets = [breast_cancer, iris, phishing, adult, avocado]
    for dataset_module in datasets:
        train, test = dataset_module.load_data()
        model = dataset_module.load_fitted_model()

        assert_that(model.predict(train.features_columns.iloc[:1]), instance_of(np.ndarray))
        assert_that(model.predict(test.features_columns.iloc[:1]), instance_of(np.ndarray))
