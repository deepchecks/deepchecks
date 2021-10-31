"""Contains unit tests for the Dataset class."""
import pandas as pd

from mlchecks import Dataset
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, instance_of, equal_to, is_, calling, raises


def assert_dataset(dataset: Dataset, args):
    assert_that(dataset, instance_of(Dataset))
    if 'df' in args:
        assert_that(dataset.data.equals(args['df']), is_(True))
    if 'features' in args:
        assert_that(dataset.features(), equal_to(args['features']))
    if 'cat_features' in args:
        assert_that(dataset.cat_features(), equal_to(args['cat_features']))
    if 'label' in args:
        assert_that(dataset.label_name(), equal_to(args['label']))
    if 'use_index' in args and args['use_index']:
        assert_that(dataset.index_col().equals(pd.Series(args['df'].index)), is_(True))
    if 'index' in args:
        assert_that(dataset.index_name(), equal_to(args['index']))
    if 'date' in args:
        assert_that(dataset.date_name(), equal_to(args['date']))
    if 'date_unit_type' in args:
        pass  # TODO
    if '_convert_date' in args:
        pass  # TODO: compare date col
    if 'max_categorical_ratio' in args:
        pass  # TODO: how to test this?
    if 'max_categories' in args:
        pass  # TODO


def test_dataset_empty_df(empty_df):
    args = {'df': empty_df}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_feature_columns(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)']}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_less_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)']}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_bad_feature(iris):
    args = {'df': iris,
            'features': ['sepal length - no exists']}
    assert_that(calling(Dataset).with_args(**args),
                raises(MLChecksValueError, 'Features must be names of columns in dataframe. Features '
                                           '{\'sepal length - no exists\'} have not been found in input dataframe.'))


def test_dataset_empty_features(iris):
    args = {'df': iris,
            'features': []}
    dataset = Dataset(**args)
    assert_that(dataset.features(), equal_to(list(iris.columns)))


def test_dataset_cat_features(diabetes_df):
    args = {'df': diabetes_df,
            'cat_features': ['sex']}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_bad_cat_feature(diabetes_df):
    args = {'df': diabetes_df,
            'cat_features': ['something else']}
    assert_that(calling(Dataset).with_args(**args),
                raises(MLChecksValueError, 'Categorical features must be a subset of features. '
                                           'Categorical features {\'something else\'} '
                                           'have not been found in feature list.'))


def test_dataset_cat_feature_not_in_features(diabetes_df):
    args = {'df': diabetes_df,
            'features': ['age',
                         'bmi',
                         'bp',
                         's1',
                         's2',
                         's3',
                         's4',
                         's5',
                         's6'],
            'cat_features': ['sex']}
    assert_that(calling(Dataset).with_args(**args),
                raises(MLChecksValueError, 'Categorical features must be a subset of features. '
                                           'Categorical features {\'sex\'} '
                                           'have not been found in feature list.'))


def test_dataset_infer_cat_features(diabetes_df):
    args = {'df': diabetes_df,
            'features': ['age',
                         'bmi',
                         'sex',
                         'bp',
                         's1',
                         's2',
                         's3',
                         's4',
                         's5',
                         's6']}

    dataset = Dataset(**args)
    args['cat_features'] = ['age', 'sex', 'bp', 's3', 's4', 's6']
    assert_dataset(dataset, args)


def test_dataset_infer_cat_features_max_categoreis(diabetes_df):
    args = {'df': diabetes_df,
            'features': ['age',
                         'bmi',
                         'sex',
                         'bp',
                         's1',
                         's2',
                         's3',
                         's4',
                         's5',
                         's6'],
            'max_categories': 10}

    dataset = Dataset(**args)
    args['cat_features'] = ['sex']
    assert_dataset(dataset, args)


def test_dataset_infer_cat_features_max_categorical_ratio(diabetes_df):
    args = {'df': diabetes_df,
            'features': ['age',
                         'bmi',
                         'sex',
                         'bp',
                         's1',
                         's2',
                         's3',
                         's4',
                         's5',
                         's6'],
            'max_categories': 10,
            'max_categorical_ratio': 0.13}

    dataset = Dataset(**args)
    args['cat_features'] = ['sex', 's6']
    assert_dataset(dataset, args)


def test_dataset_label(iris):
    args = {'df': iris,
            'label': 'target'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_label_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'label': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(MLChecksValueError, 'label can not be part of the features'))


def test_dataset_use_index(iris):
    args = {'df': iris,
            'use_index': True}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_use_index(iris):
    args = {'df': iris,
            'index': 'target',
            'use_index': True}
    assert_that(calling(Dataset).with_args(**args),
                raises(MLChecksValueError, 'parameter use_index cannot be True if index is given'))


def test_dataset_index_from_column(iris):
    args = {'df': iris,
            'label': 'target'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_in_df(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)'],
            'index': 'index'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)



'''
TODO: tests
    features is lable/index/date

max_categoreis/ max_categorical_ratio is confusing
'''
