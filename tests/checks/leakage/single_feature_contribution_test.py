"""
Contains unit tests for the dataset_info check
"""
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks.leakage.single_feature_contribution import single_feature_contribution
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_assert_single_feature_contribution():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['x4'] = df['x1'] * 0.5 + df['x2']
    df['label'] = df['x2'] + 0.1 * df['x1']

    result = single_feature_contribution(dataset=Dataset(df))

    assert_that(result.value, equal_to({'x2': 0.8410436710134066, 'x4': 0.5196251242216743, 'x1': 0.0, 'x3': 0.0}))

#
# def test_dataset_wrong_input():
#     X = "wrong_input"
#     assert_that(calling(dataset_info).with_args(X),
#                 raises(MLChecksValueError, 'dataset_info check must receive a DataFrame or a Dataset object'))
#
#
# def test_dataset_info_object(iris_dataset):
#     di = DatasetInfo()
#     result = di.run(iris_dataset, model=None)
#     assert_that(result.value, equal_to((150, 5)))
#
#
# def test_dataset_info_dataframe(iris):
#     result = dataset_info(iris)
#     assert_that(result.value, equal_to((150, 5)))