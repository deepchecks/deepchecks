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
"""Contains unit tests for the Dataset class."""
import typing as t
import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from hamcrest import (
    assert_that, instance_of, equal_to, is_,
    calling, raises, not_none, has_property, all_of, contains_exactly, has_item, has_length
)

from deepchecks.tabular import Dataset
from deepchecks.utils.validation import ensure_dataframe_type
from deepchecks.core.errors import DeepchecksValueError, DatasetValidationError


def assert_dataset(dataset: Dataset, args):
    assert_that(dataset, instance_of(Dataset))
    if 'df' in args:
        assert_that(dataset.data.equals(args['df']), is_(True))
    if 'features' in args:
        assert_that(dataset.features, equal_to(args['features']))
        if 'df' in args:
            assert_that(dataset.data[dataset.features].equals(args['df'][args['features']]), is_(True))
    if 'cat_features' in args:
        assert_that(dataset.cat_features, equal_to(args['cat_features']))
    if 'label_name' in args:
        assert_that(dataset.label_name, equal_to(args['label_name']))
        assert_that(dataset.data[dataset.label_name].equals(pd.Series(args['df'][args['label_name']])), is_(True))
    if 'use_index' in args and args['use_index']:
        assert_that(dataset.index_col.equals(pd.Series(args['df'].index)), is_(True))
    if 'index_name' in args:
        assert_that(dataset.index_name, equal_to(args['index_name']))
        if 'set_index_from_dataframe_index' not in args or not args['set_index_from_dataframe_index']:
            assert_that(dataset.index_col.equals(pd.Series(dataset.data[args['index_name']])), is_(True))
        elif args['index_name']:
            assert_that(dataset.index_col.equals(
                pd.Series(args['df'].index.get_level_values(args['index_name']),
                          name=args['df'].index.name,
                          index=args['df'].index)
            ),
                is_(True)
            )
        else:
            assert_that(dataset.index_col.equals(
                pd.Series(args['df'].index.get_level_values(0),
                          name=args['df'].index.name,
                          index=args['df'].index)
            ),
                is_(True)
            )
    if 'datetime_name' in args:
        assert_that(dataset.datetime_name, equal_to(args['datetime_name']))
        if 'set_datetime_from_dataframe_index' not in args or not args['set_datetime_from_dataframe_index']:
            assert_that(dataset.datetime_col.equals(pd.Series(dataset.data[args['datetime_name']])), is_(True))
        elif args['datetime_name']:
            assert_that(dataset.datetime_col.equals(
                pd.Series(args['df'].index.get_level_values(args['datetime_name']),
                          name='datetime',
                          index=args['df'].index)
            ),
                is_(True)
            )
        else:
            assert_that(dataset.datetime_col.equals(
                pd.Series(args['df'].index.get_level_values(0),
                          name='datetime',
                          index=args['df'].index)
            ),
                is_(True)
            )

def test_that_mutable_properties_modification_does_not_affect_dataset_state(iris):
    dataset = Dataset(
        df=iris,
        features=[
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ]
    )

    features = dataset.features
    cat_features = dataset.cat_features

    features.append("New value")
    cat_features.append("New value")

    assert_that("New value" not in dataset.features)
    assert_that("New value" not in dataset.cat_features)


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
    assert_that(dataset.features, equal_to(list([])))


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
            'label': 'target'}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_label_name_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'label': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'label column target can not be a feature column'))


def test_dataset_bad_label_name(iris):
    args = {'df': iris,
            'label': 'shmabel'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'label column shmabel not found in dataset columns'))


def test_dataset_use_index(iris):
    args = {'df': iris,
            'set_index_from_dataframe_index': True}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_use_index_by_name(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'index_name': 'first',
            'set_index_from_dataframe_index': True}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_use_index_by_non_existent_name(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'index_name': 'third',
            'set_index_from_dataframe_index': True}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'Index third not found in dataframe index level names.')
                )


def test_dataset_index_use_index_by_int(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'index_name': 0,
            'set_index_from_dataframe_index': True}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_index_use_index_by_int_too_large(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'index_name': 2,
            'set_index_from_dataframe_index': True}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'Dataframe index has less levels than 3.')
                )


def test_dataset_date_use_date_by_int_too_large(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'datetime_name': 2,
            'set_datetime_from_dataframe_index': True}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'Dataframe index has less levels than 3.')
                )


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
                raises(DeepchecksValueError, 'Index column index not found in dataset columns. If you attempted to use '
                                             'the dataframe index, set set_index_from_dataframe_index to True instead.')
                )


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
    args = {'datetime_name': 'target'}
    dataset = Dataset(iris, **args)
    assert_dataset(dataset, args)


def test_dataset_date_not_in_columns(iris):
    args = {'datetime_name': 'date'}
    assert_that(calling(Dataset).with_args(iris, **args),
                raises(DeepchecksValueError,
                       'Datetime column date not found in dataset columns. If you attempted to use the dataframe index,'
                       ' set set_datetime_from_dataframe_index to True instead.'))


def test_dataset_date_in_features(iris):
    args = {'df': iris,
            'features': ['sepal length (cm)',
                         'sepal width (cm)',
                         'petal length (cm)',
                         'petal width (cm)',
                         'target'],
            'datetime_name': 'target'}
    assert_that(calling(Dataset).with_args(**args),
                raises(DeepchecksValueError, 'datetime column target can not be a feature column'))


def test_dataset_datetime_use_datetime_by_name(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'datetime_name': 'first',
            'set_datetime_from_dataframe_index': True,
            'convert_datetime': False}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_datetime_use_datetime_by_int(multi_index_dataframe):
    args = {'df': multi_index_dataframe,
            'datetime_name': 0,
            'set_datetime_from_dataframe_index': True,
            'convert_datetime': False}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)


def test_dataset_date_args_single_arg():
    df = pd.DataFrame({'date': [1, 2]})
    args = {'datetime_name': 'date',
            'datetime_args': {'unit': 'D'}}
    dataset = Dataset(df, **args)
    assert_dataset(dataset, args)
    date_col = dataset.datetime_col
    assert_that(date_col, not_none())
    # disable false positive
    # pylint:disable=unsubscriptable-object
    assert_that(date_col[0], is_(pd.to_datetime(1, unit='D')))
    assert_that(date_col[1], is_(pd.to_datetime(2, unit='D')))


def test_dataset_date_args_multi_arg():
    df = pd.DataFrame({'date': [160, 180]})
    args = {'datetime_name': 'date',
            'datetime_args': {'unit': 'D', 'origin': '2020-02-01'}}
    dataset = Dataset(df, **args)
    assert_dataset(dataset, args)
    date_col = dataset.datetime_col
    assert_that(date_col, not_none())
    # disable false positive
    # pylint:disable=unsubscriptable-object
    assert_that(date_col[0], is_(pd.to_datetime(160, unit='D', origin='2020-02-01')))
    assert_that(date_col[1], is_(pd.to_datetime(180, unit='D', origin='2020-02-01')))


def test_dataset_date_convert_date():
    df = pd.DataFrame({'date': [1, 2]})
    args = {'df': df,
            'datetime_name': 'date',
            'convert_datetime': False}
    dataset = Dataset(**args)
    assert_dataset(dataset, args)
    date_col = dataset.datetime_col
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


def test_dataset_select_method(iris):
    dataset = Dataset(iris)
    filtered = dataset.select(columns=['target'])
    assert_that(filtered, instance_of(Dataset))


def test_dataset_select_ignore_columns(iris):
    dataset = Dataset(iris)
    filtered = dataset.select(ignore_columns=['target'])
    assert_that(filtered, instance_of(Dataset))


def test_dataset_select_same_table(iris):
    dataset = Dataset(iris, features=['target'])
    filtered = dataset.select(ignore_columns=['target'])
    assert_that(filtered, instance_of(Dataset))


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
        calling(Dataset.from_numpy).with_args(iris.data, iris.target, columns=['X1', ]),
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

    features = dataset.data[dataset.features]
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
        labels = dataset.data[dataset.label_name]
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
        features=[0, 1, 2],
        label=3,
        cat_features=[0],
    )

    assert_that(actual=dataset.features, matcher=contains_exactly(0, 1, 2))
    assert_that(actual=dataset.label_name, matcher=equal_to(3))
    assert_that(actual=dataset.cat_features, matcher=contains_exactly(0))

    assert_that(
        (dataset.data[dataset.features] == df.drop(3, axis=1))
            .all().all()
    )
    assert_that(
        (dataset.data[dataset.label_name] == df[3]).all()
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
    label = iris['target'].rename('actual')
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'actual'))


def test_train_test_split(iris):
    # Arrange
    label = iris['target'].rename('actual')
    data = iris.drop('target', axis=1)
    dataset = Dataset(data, label)
    # Act
    train_ds, test_ds = dataset.train_test_split()
    # Assert
    assert_that(train_ds.n_samples, 150)
    assert_that(test_ds.n_samples, 50)


def test_train_test_split_changed(iris):
    # Arrange
    label = iris['target'].rename('actual')
    data = iris.drop('target', axis=1)
    dataset = Dataset(data, label)
    # Act
    train_ds, test_ds = dataset.train_test_split(train_size=0.2, test_size=0.1)
    # Assert
    assert_that(train_ds.n_samples, 15)
    assert_that(test_ds.n_samples, 10)


def test_inferred_label_type_cat(diabetes_df):
    # Arrange
    label = diabetes_df['target'].rename('actual')
    data = diabetes_df.drop('target', axis=1)
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.label_type, is_('regression_label'))


def test_inferred_label_type_reg(iris):
    # Arrange
    label = iris['target'].rename('actual')
    data = iris.drop('target', axis=1)
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.label_type, is_('classification_label'))


def test_set_label_type(iris):
    # Arrange
    label = iris['target'].rename('actual')
    data = iris.drop('target', axis=1)
    dataset = Dataset(data, label, label_type='regression_label')
    # Assert
    assert_that(dataset.label_type, is_('regression_label'))


def test_label_series_name_already_exists(iris):
    # Arrange
    label = iris['target']
    data = iris.drop('target', axis=1)
    label = label.rename(iris.columns[0])

    # Act & Assert
    assert_that(calling(Dataset).with_args(data, label=label),
                raises(DeepchecksValueError, r'Data has column with name "sepal length \(cm\)", use pandas rename to '
                                             r'change label name or remove the column from the dataframe'))


def test_label_series_without_name_default_name_exists(iris):
    # Arrange
    label = pd.Series([0] * len(iris))

    # Act & Assert
    assert_that(iris.columns, has_item('target'))
    assert_that(calling(Dataset).with_args(iris, label=label),
                raises(DeepchecksValueError, r'Can\'t set default label name "target" since it already exists in the '
                                             r'dataframe\. use pandas name parameter to give the label a unique name'))


def test_label_is_numpy_array(iris):
    # Arrange
    label = np.ones(len(iris))
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'target'))


def test_label_is_numpy_column(iris):
    # Arrange
    label = np.ones((len(iris), 1))
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'target'))


def test_label_is_numpy_row(iris):
    # Arrange
    label = np.ones((1, len(iris)))
    data = iris.drop('target', axis=1)
    # Act
    dataset = Dataset(data, label)
    # Assert
    assert_that(dataset.features, equal_to(list(data.columns)))
    assert_that(dataset.data.columns, contains_exactly(*data.columns, 'target'))


def test_label_is_dataframe(iris):
    # Arrange
    label = pd.DataFrame(data={'actual': [0] * len(iris)})
    # Act
    dataset = Dataset(iris, label)
    # Assert
    assert_that(dataset.features, equal_to(list(iris.columns)))
    assert_that(dataset.data.columns, contains_exactly(*iris.columns, 'actual'))


def test_label_unsupported_type(iris):
    # Arrange
    label = {}

    # Act & Assert
    assert_that(calling(Dataset).with_args(iris, label=label),
                raises(DeepchecksValueError, 'Unsupported type for label: dict'))


def test_label_dataframe_with_multi_columns(iris):
    # Arrange
    label = pd.DataFrame(data={'col1': [0] * len(iris), 'col2': [1] * len(iris)})

    # Act & Assert
    assert_that(calling(Dataset).with_args(iris, label=label),
                raises(DeepchecksValueError, 'Label must have a single column'))


def test_label_numpy_multi_2d_array(iris):
    # Arrange
    label = np.ones((3, len(iris)))
    iris = iris.drop('target', axis=1)
    # Act & Assert
    assert_that(calling(Dataset).with_args(iris, label=label),
                raises(DeepchecksValueError, 'Label must be either column vector or row vector'))


def test_label_numpy_multi_4d_array(iris):
    # Arrange
    label = np.ones((1, 1, 1, len(iris)))
    iris = iris.drop('target', axis=1)
    # Act & Assert
    assert_that(calling(Dataset).with_args(iris, label=label),
                raises(DeepchecksValueError, 'Label must be either column vector or row vector'))


def test_sample_with_nan_labels(iris):
    # Arrange
    iris = iris.copy()
    iris.loc[iris['target'] != 2, 'target'] = None
    dataset = Dataset(iris, label='target')
    # Act
    sample = dataset.sample(10000)
    # Assert
    assert_that(sample, has_length(150))


def test_sample_drop_nan_labels(iris):
    # Arrange
    iris = iris.copy()
    iris.loc[iris['target'] != 2, 'target'] = None
    dataset = Dataset(iris, label='target')
    # Act
    sample = dataset.sample(10000, drop_na_label=True)
    # Assert
    assert_that(sample, has_length(50))


def test__ensure_not_empty_dataset(iris: pd.DataFrame):
    # Arrange
    ds = Dataset(iris)
    # Act
    ds = Dataset.ensure_not_empty_dataset(ds)


def test__ensure_not_empty_dataset__with_empty_dataset():
    # Arrange
    ds = Dataset(pd.DataFrame())
    # Assert
    assert_that(
        calling(Dataset.ensure_not_empty_dataset).with_args(ds),
        raises(DatasetValidationError, r'dataset cannot be empty')
    )


def test__ensure_not_empty_dataset__with_dataframe(iris: pd.DataFrame):
    # Arrange
    ds = Dataset.ensure_not_empty_dataset(iris)
    # Assert
    assert_that(ds, instance_of(Dataset))
    assert_that(ds.features, has_length(0))
    assert_that(ds.label_name, equal_to(None))
    assert_that(ds.n_samples, equal_to(len(iris)))


def test__ensure_not_empty_dataset__with_empty_dataframe():
    # Assert
    assert_that(
        calling(Dataset.ensure_not_empty_dataset).with_args(pd.DataFrame()),
        raises(DatasetValidationError, r'dataset cannot be empty')
    )


def test__datasets_share_features(iris: pd.DataFrame):
    # Arrange
    ds = Dataset(iris)
    # Assert
    assert_that(
        Dataset.datasets_share_features(ds, ds),
        equal_to(True)
    )


def test__datasets_share_features__with_features_lists_ordered_differently():
    # Arrange
    # Features are the same, but their order in the lists is different
    # that must not have affect on the outcome
    first_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X1', 'X2'])
    second_ds = Dataset(random_classification_dataframe(), cat_features=['X2', 'X0', 'X1'])
    # Assert
    assert_that(
        Dataset.datasets_share_features(first_ds, second_ds),
        equal_to(True)
    )


def test__datasets_share_features__with_wrong_args(iris: pd.DataFrame):
    # Arrange
    ds = Dataset(iris, label="target")
    # Assert
    assert_that(
        calling(Dataset.datasets_share_features).with_args([ds]),
        raises(AssertionError, r"'datasets' must contains at least two items")
    )


def test__datasets_share_features__when_it_must_return_false(
    iris: pd.DataFrame,
    diabetes_df: pd.DataFrame
):
    # Arrange
    iris_ds = Dataset(iris)
    diabetes_ds = Dataset(diabetes_df)
    # Assert
    assert_that(
        Dataset.datasets_share_features(iris_ds, diabetes_ds),
        equal_to(False)
    )

def test__datasets_share_features__with_differently_ordered_datasets_list(
    iris: pd.DataFrame,
    diabetes_df: pd.DataFrame
):
    # Arrange
    iris_ds = Dataset(iris)
    diabetes_ds = Dataset(diabetes_df)

    # Assert
    # no matter in which order datasets are placed in the list
    # outcome must be the same
    assert_that(
        Dataset.datasets_share_features(iris_ds, diabetes_ds),
        equal_to(False)
    )
    assert_that(
        Dataset.datasets_share_features(diabetes_ds, iris_ds),
        equal_to(False)
    )


def test__datasets_share_categorical_features(diabetes_df: pd.DataFrame):
    # Arrange
    ds = Dataset(diabetes_df, cat_features=['sex'])
    # Assert
    assert_that(
        Dataset.datasets_share_categorical_features(ds, ds),
        equal_to(True)
    )


def test__datasets_share_categorical_features__with_features_lists_ordered_differently():
    # Arrange
    # Features are the same, but their order in the lists is different
    # that must not have affect on the outcome
    first_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X1', 'X2'])
    second_ds = Dataset(random_classification_dataframe(), cat_features=['X2', 'X0', 'X1'])
    # Assert
    assert_that(
        Dataset.datasets_share_categorical_features(first_ds, second_ds),
        equal_to(True)
    )


def test__datasets_share_categorical_features__with_wrong_args(diabetes_df: pd.DataFrame):
    # Arrange
    ds = Dataset(diabetes_df, cat_features=['sex'])
    # Assert
    assert_that(
        calling(Dataset.datasets_share_categorical_features).with_args(ds),
        raises(AssertionError, r"'datasets' must contains at least two items")
    )


def test__datasets_share_categorical_features__when_it_must_return_false():
    # Arrange
    first_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X1'])
    second_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X2', 'X3'])
    # Assert
    assert_that(
        Dataset.datasets_share_categorical_features(first_ds, second_ds),
        equal_to(False)
    )


def test__datasets_share_categorical_features__with_differently_ordered_datasets_list():
    # Arrange
    first_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X1'])
    second_ds = Dataset(random_classification_dataframe(), cat_features=['X0', 'X2', 'X3'])

    # Assert
    # no matter in which order datasets are placed in the list
    # outcome must be the same
    assert_that(
        Dataset.datasets_share_categorical_features(first_ds, second_ds),
        equal_to(False)
    )
    assert_that(
        Dataset.datasets_share_categorical_features(first_ds, second_ds),
        equal_to(False)
    )


def test__datasets_share_label():
    ds = Dataset(random_classification_dataframe(), label="target")
    assert_that(
        Dataset.datasets_share_label(ds, ds),
        equal_to(True)
    )


def test__datasets_share_label__with_wrong_args(iris: pd.DataFrame):
    # Arrange
    ds = Dataset(iris, label="target")
    # Assert
    assert_that(
        calling(Dataset.datasets_share_label).with_args([ds]),
        raises(AssertionError, r"'datasets' must contains at least two items")
    )


def test__datasets_share_label__when_it_must_return_false(iris: pd.DataFrame):
    # Arrange
    df = random_classification_dataframe()
    df.rename(columns={"target": "Y_target"}, inplace=True)
    ds = Dataset(df, label="Y_target")
    iris_ds = Dataset(iris, label="target")
    # Assert
    assert_that(
        Dataset.datasets_share_label(ds, iris_ds),
        equal_to(False)
    )


def test__datasets_share_label__with_differently_ordered_datasets_list(iris: pd.DataFrame):
    # Arrange
    df = random_classification_dataframe()
    df.rename(columns={"target": "Y_target"}, inplace=True)
    ds = Dataset(df, label="Y_target")
    iris_ds = Dataset(iris, label="target")
    # Assert
    assert_that(
        Dataset.datasets_share_label(ds, iris_ds),
        equal_to(False)
    )
    assert_that(
        Dataset.datasets_share_label(iris_ds, ds),
        equal_to(False)
    )


def random_classification_dataframe(n_samples=100, n_features=5) -> pd.DataFrame:
    x, y, *_ = make_classification(n_samples=n_samples, n_features=n_features)
    df = pd.DataFrame(x,columns=[f'X{i}'for i in range(n_features)])
    df['target'] = y
    return df