# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains unit tests for the Dataset class."""
import typing as t
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from deepchecks import Dataset, ensure_dataframe_type
from deepchecks.errors import DeepchecksValueError
from hamcrest import (
    assert_that, instance_of, equal_to, is_,
    calling, raises, not_none, has_property, all_of, contains_exactly
)


def assert_dataset(dataset: Dataset, args):
    assert_that(dataset, instance_of(Dataset))
    if 'df' in args:
        assert_that(dataset.data.equals(args['df']), is_(True))
    if 'features' in args:
        assert_that(dataset.features, equal_to(args['features']))
        if 'df' in args:
            assert_that(dataset.features_columns.equals(args['df'][args['features']]), is_(True))
    if 'cat_features' in args:
        assert_that(dataset.cat_features, equal_to(args['cat_features']))
    if 'label_name' in args:
        assert_that(dataset.label_name, equal_to(args['label_name']))
        assert_that(dataset.label_col.equals(pd.Series(args['df'][args['label_name']])), is_(True))
    if 'use_index' in args and args['use_index']:
        assert_that(dataset.index_col.equals(pd.Series(args['df'].index)), is_(True))
    if 'index_name' in args:
        assert_that(dataset.index_name, equal_to(args['index_name']))
        assert_that(dataset.index_col.equals(pd.Series(args['df'][args['index_name']])), is_(True))
    if 'date_name' in args:
        assert_that(dataset.date_name, equal_to(args['date_name']))
        if ('convert_date_' in args) and (args['convert_date_'] is False):
            assert_that(dataset.date_col.equals(pd.Series(args['df'][args['date_name']])), is_(True))
        else:
            for date in dataset.date_col:
                assert_that(date, instance_of(pd.Timestamp))


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
                raises(DeepchecksValueError, 'Features must be names of columns in dataframe. Features '
                                           '{\'sepal length - no exists\'} have not been found in input dataframe.'))


def test_dataset_empty_features(iris):
    args = {'df': iris,
            'features': []}
    dataset = Dataset(**args)
    assert_that(dataset.features, equal_to(list(iris.columns)))


def test_dataset_cat_features(diabetes_df):
    args = {'df': diabetes_df,
            'cat_features': ['sex']}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_bad_cat_feature(diabetes_df):
    args = {'df': diabetes_df,
            'cat_features': ['something else']}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'Categorical features must be a subset of features. '
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
                raises(DeepchecksValueError, 'Categorical features must be a subset of features. '
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
    args['cat_features'] = ['sex']
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
            'max_categories': 60,
            'max_float_categories': 60}

    dataset = Dataset(**args)
    args['cat_features'] = ['age', 'sex', 's6']
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
    args['cat_features'] = ['sex']
    assert_dataset(dataset, args)


def test_dataset_label_name(iris):
    args = {'df': iris,
            'label_name': 'target'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_label_name_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'label_name': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'label column target can not be a feature column'))


def test_dataset_bad_label_name(iris):
    args = {'df': iris,
            'label_name': 'shmabel'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'label column shmabel not found in dataset columns'))


def test_dataset_use_index(iris):
    args = {'df': iris,
            'create_index_from_dataframe_index': True}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_use_index(iris):
    args = {'df': iris,
            'index_name': 'target',
            'create_index_from_dataframe_index': True}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'parameter create_index_from_dataframe_index cannot be True if index is given'))


def test_dataset_index_from_column(iris):
    args = {'df': iris,
            'index_name': 'target'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_in_df(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)'],
            'index_name': 'index'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'index column index not found in dataset columns. If you attempted to use '
                                           'the dataframe index, set create_index_from_dataframe_index to True instead.'))


def test_dataset_index_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'index_name': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'index column target can not be a feature column'))


def test_dataset_date(iris):
    args = {'date_name': 'target'}
    dataset = Dataset(iris, **args)
    assert_dataset(dataset, args)


def test_dataset_date_not_in_columns(iris):
    args = {'date_name': 'date'}
    assert_that(calling(Dataset).with_args(iris, **args),
                raises(DeepchecksValueError, 'date column date not found in dataset columns'))


def test_dataset_date_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'date_name': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'date column target can not be a feature column'))


def test_dataset_date_args_single_arg():
    df = pd.DataFrame({'date': [1, 2]})
    args = {'date_name': 'date',
            'date_args': {'unit': 'D'}}
    dataset = Dataset(df, **args)
    assert_dataset(dataset, args)
    date_col = dataset.date_col
    assert_that(date_col, not_none())
    # disable false positive
    # pylint:disable=unsubscriptable-object
    assert_that(date_col[0], is_(pd.to_datetime(1, unit='D')))
    assert_that(date_col[1], is_(pd.to_datetime(2, unit='D')))


def test_dataset_date_args_multi_arg():
    df = pd.DataFrame({'date': [160, 180]})
    args = {'date_name': 'date',
            'date_args': {'unit': 'D', 'origin': '2020-02-01'}}
    dataset = Dataset(df, **args)
    assert_dataset(dataset, args)
    date_col = dataset.date_col
    assert_that(date_col, not_none())
    # disable false positive
    # pylint:disable=unsubscriptable-object
    assert_that(date_col[0], is_(pd.to_datetime(160, unit='D', origin='2020-02-01')))
    assert_that(date_col[1], is_(pd.to_datetime(180, unit='D', origin='2020-02-01')))


def test_dataset_date_convert_date():
    df = pd.DataFrame({'date': [1, 2]})
    args = {'df': df,
            'date_name': 'date',
            'convert_date_': False}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)
    date_col = dataset.date_col
    assert_that(date_col, not_none())
    # disable false positive
    # pylint:disable=unsubscriptable-object
    assert_that(date_col[0], is_(1))
    assert_that(date_col[1], is_(2))


def test_dataset_data(iris):
    dataset = Dataset(iris)
    iris.equals(dataset.data)


def test_dataset_n_samples(iris):
    dataset = Dataset(iris)
    assert_that(dataset.n_samples, is_(iris.shape[0]))


def test_dataset_no_index_col(iris):
    dataset = Dataset(iris)
    assert_that(dataset.index_col, is_(None))


def test_dataset_validate_label(iris):
    dataset = Dataset(iris, label_name='target')
    dataset.validate_label()


def test_dataset_validate_no_label(iris):
    dataset = Dataset(iris)
    assert_that(calling(dataset.validate_label),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_dataset_validate_date(iris):
    dataset = Dataset(iris, date_name='target')
    dataset.validate_date()


def test_dataset_validate_no_date(iris):
    dataset = Dataset(iris)
    assert_that(calling(dataset.validate_date),
                raises(DeepchecksValueError, 'Check requires dataset to have a date column'))


def test_dataset_validate_index(iris):
    dataset = Dataset(iris, index_name='target')
    dataset.validate_index()


def test_dataset_validate_no_index(iris):
    dataset = Dataset(iris)
    assert_that(calling(dataset.validate_index),
                raises(DeepchecksValueError, 'Check requires dataset to have an index column'))


def test_dataset_filter_columns_with_validation(iris):
    dataset = Dataset(iris)
    filtered = dataset.filter_columns_with_validation(columns=['target'])
    assert_that(filtered, instance_of(Dataset))


def test_dataset_filter_columns_with_validation_ignore_columns(iris):
    dataset = Dataset(iris)
    filtered = dataset.filter_columns_with_validation(ignore_columns=['target'])
    assert_that(filtered, instance_of(Dataset))


def test_dataset_filter_columns_with_validation_same_table(iris):
    dataset = Dataset(iris, features=['target'])
    filtered = dataset.filter_columns_with_validation(ignore_columns=['target'])
    assert_that(filtered, instance_of(Dataset))


def test_dataset_validate_shared_features(diabetes):
    train, val = diabetes
    assert_that(train.validate_shared_features(val), is_(val.features))


def test_dataset_validate_shared_features_fail(diabetes, iris_dataset):
    train = diabetes[0]
    assert_that(calling(train.validate_shared_features).with_args(iris_dataset),
                raises(DeepchecksValueError, 'Check requires datasets to share the same features'))


def test_dataset_validate_shared_label(diabetes):
    train, val = diabetes
    assert_that(train.validate_shared_label(val), is_(val.label_name))


def test_dataset_validate_shared_labels_fail(diabetes, iris_dataset):
    train = diabetes[0]
    assert_that(calling(train.validate_shared_label).with_args(iris_dataset),
                raises(DeepchecksValueError, 'Check requires datasets to share the same label'))


def test_dataset_shared_categorical_features(diabetes_df, iris):
    diabetes_dataset = Dataset(diabetes_df)
    iris_dataset = Dataset(iris)
    assert_that(calling(diabetes_dataset.validate_shared_categorical_features).with_args(iris_dataset),
                raises(DeepchecksValueError, 'Check requires datasets to share'
                                           ' the same categorical features'))


def test_validate_dataset_or_dataframe_empty_df(empty_df):
    assert_that(calling(Dataset.validate_dataset_or_dataframe).with_args(empty_df),
                raises(DeepchecksValueError, 'dataset cannot be empty'))


def test_validate_dataset_or_dataframe_empty_dataset(empty_df):
    assert_that(calling(Dataset.validate_dataset_or_dataframe).with_args(Dataset(empty_df)),
                raises(DeepchecksValueError, 'dataset cannot be empty'))


def test_validate_dataset_or_dataframe(iris):
    assert_that(Dataset.validate_dataset_or_dataframe(iris), Dataset(iris))


def test_validate_dataset_empty_df(empty_df):
    assert_that(calling(Dataset.validate_dataset).with_args(Dataset(empty_df)),
                raises(DeepchecksValueError, 'Check requires a non-empty dataset'))


def test_validate_dataset_not_dataset():
    assert_that(calling(Dataset.validate_dataset).with_args('not_dataset'),
                raises(DeepchecksValueError, 'Check requires dataset to be of type Dataset. instead got:'
                                           ' str'))


def test_ensure_dataframe_type(iris):
    assert_that(ensure_dataframe_type(iris).equals(iris), is_(True))


def test_ensure_dataframe_type_dataset(iris):
    assert_that(ensure_dataframe_type(Dataset(iris)).equals(iris), is_(True))


def test_ensure_dataframe_type_fail():
    assert_that(calling(ensure_dataframe_type).with_args('not dataset'),
                raises(DeepchecksValueError, 'dataset must be of type DataFrame or Dataset, but got: str'))


def test_dataset_initialization_from_numpy_arrays():
    iris = load_iris()
    validate_dataset_created_from_numpy_arrays(
        dataset=Dataset.from_numpy(iris.data, iris.target),
        features_array=iris.data,
        labels_array=iris.target,
    )


def test_dataset_initialization_from_numpy_arrays_with_specified_features_names():
    iris = load_iris()
    label_column_name = 'label-col'
    feature_columns_names = ['feature-1', 'feature-2', 'feature-3', 'feature-4']

    ds = Dataset.from_numpy(
        iris.data, iris.target,
        label_name=label_column_name,
        columns=feature_columns_names
    )
    validate_dataset_created_from_numpy_arrays(
        dataset=ds,
        features_array=iris.data,
        labels_array=iris.target,
        feature_columns_names=feature_columns_names,
        label_column_name=label_column_name
    )


def test_dataset_of_features_initialization_from_numpy_array():
    iris = load_iris()
    validate_dataset_created_from_numpy_arrays(
        dataset=Dataset.from_numpy(iris.data),
        features_array=iris.data
    )


def test_dataset_initialization_from_numpy_arrays_of_different_length():
    iris = load_iris()
    assert_that(
        calling(Dataset.from_numpy).with_args(iris.data, iris.target[:10]),
        raises(
            DeepchecksValueError,
            "Number of samples of label and data must be equal"
        )
    )


def test_dataset_of_features_initialization_from_not_2d_numpy_arrays():
    iris = load_iris()
    assert_that(
        calling(Dataset.from_numpy).with_args(iris.target),
        raises(
            DeepchecksValueError,
            r"'from_numpy' constructor expecting columns \(args\[0\]\) to be not empty two dimensional array\."
        )
    )


def test_dataset_initialization_from_numpy_arrays_without_providing_args():
    assert_that(
        calling(Dataset.from_numpy).with_args(),
        raises(
            DeepchecksValueError,
            r"'from_numpy' constructor expecting to receive two numpy arrays \(or at least one\)\."
            r"First array must contains the columns and second the labels\."
        )
    )


def test_dataset_initialization_from_numpy_arrays_with_wrong_number_of_feature_columns_names():
    iris = load_iris()
    assert_that(
        calling(Dataset.from_numpy).with_args(iris.data, iris.target, columns=['X1',]),
        raises(
            DeepchecksValueError,
            '4 columns were provided '
            r'but only 1 name\(s\) for them`s.'
        )
    )

def test_dataset_initialization_from_numpy_empty_arrays():
    iris = load_iris()
    assert_that(
        calling(Dataset.from_numpy).with_args(iris.data[:0], iris.target),
        raises(
            DeepchecksValueError,
            r"'from_numpy' constructor expecting columns \(args\[0\]\) "
            r"to be not empty two dimensional array\."
        )
    )


def validate_dataset_created_from_numpy_arrays(
    dataset: Dataset,
    features_array: np.ndarray,
    labels_array: np.ndarray = None,
    feature_columns_names: t.Sequence[str] = None,
    label_column_name: str = 'target'
):
    if feature_columns_names is None:
        feature_columns_names = [str(index) for index in range(1, features_array.shape[1] + 1)]

    features = dataset.features_columns
    feature_names = dataset.features

    assert_that(features, all_of(
        instance_of(pd.DataFrame),
        has_property('shape', equal_to(features_array.shape)),
    ))
    assert_that(all(features == features_array))
    assert_that(feature_names, all_of(
        instance_of(list),
        contains_exactly(*feature_columns_names)
    ))

    if labels_array is not None:
        labels = dataset.label_col
        label_name = dataset.label_name

        assert_that(labels, all_of(
            instance_of(pd.Series),
            has_property('shape', equal_to(labels_array.shape))
        ))
        assert_that(all(labels == labels_array))
        assert_that(label_name, equal_to(label_column_name))


def test_dataset_initialization_with_integer_columns():
    df = pd.DataFrame.from_records([
        {0: 'a', 1: 0.6, 2: 0.7, 3: 1},
        {0: 'b', 1: 0.33, 2: 0.14, 3: 0},
        {0: 'c', 1: 0.24, 2: 0.07, 3: 0},
        {0: 'c', 1: 0.89, 2: 0.56, 3: 1}
    ])

    dataset = Dataset(
        df=df,
        features=[0,1,2],
        label_name=3,
        cat_features=[0],
    )

    assert_that(actual=dataset.features, matcher=contains_exactly(0,1,2))
    assert_that(actual=dataset.label_name, matcher=equal_to(3))
    assert_that(actual=dataset.cat_features, matcher=contains_exactly(0))

    assert_that(
        (dataset.features_columns == df.drop(3, axis=1))
        .all().all()
    )
    assert_that(
        (dataset.label_col == df[3]).all()
    )


def test_dataset_label_without_name(iris):
    # Arrange
    label = iris['target']
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'target'))


def test_dataset_label_with_name(iris):
    # Arrange
    label = iris['target']
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label, label_name='actual')
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'actual'))
