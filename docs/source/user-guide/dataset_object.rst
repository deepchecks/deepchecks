.. _dataset_object:

====================
The Dataset Object
====================
The dataset is one of the basic blocks of deepchecks. It is a container for the data and its relevant metadata, like special column names (label, date, index, etc).
Some of the checks allows to use a dataframe directly, but some others requires the metadata in order to run, so they are limited to working only with Datasets.

Class Parameters
===================
All of the parameters are optional.

.. list-table::
    :widths: 20 20 50 10
    :header-rows: 1

    * - Name
      - Type
      - Description
      - Default
    * - label
      - pandas.Series
      - Data of labels as separate series from the data
      - None
    * - features
      - List[Hashable]
      - Names of the features in the data
      - None
    * - cat_features
      - List[Hashable]
      - Names of the categorical features in the data. Must be subset of `features`
      - None
    * - label_name
      - Hashable
      - Name of label column in the data
      - None
    * - use_index
      - bool
      - If data is dataframe, whether to use the dataframe index as index column for index related checks
      - False
    * - index_name
      - Hashable
      - Name of index column in the data
      - None
    * - date_name
      - Hashable
      - Name of date column in the data
      - None
    * - date_unit_type
      - str
      - Unit to convert date column if it's numeric. using `pandas.Timestamp <https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html>`__ to convert
      - None
    * - max_categorical_ratio
      - float
      - Used to infer which columns are categorical (if `cat_features` isn't explicitly passed).
        Set maximum ratio of unique values in a column in order for it to be categorical.
        The higher the value, the chance of column inferred as categorical is higher
      - 0.01
    * - max_categories
      - int
      - Used to infer which columns are categorical (if `cat_features` isn't explicitly passed).
        Set maximum number of unique values in a column in order for it to be categorical.
        The higher the value, the chance of column inferred as categorical is higher
      - 30
    * - max_float_categories
      - int
      - Same as `max_categories` but for columns of type float
      - 5
    * - convert_date
      - bool
      - Whether to convert date column if it's numeric to date
      - True

Inferring Features And Categorical Features
================================================
Dataset defines which columns of the data are features and of them which are categorical features.
For features, if parameter `features` not passed explicitly, all will be considered features apart from the label, index and date.
For categorical features, if parameter `cat_features` not passed explicitly, the following logic runs on every column to determine
whether the column is categorical or not:

* if columns is float type:
    * number of unique < `max_float_categories`
* else:
    *  number of unique < `max_categories` AND (number of unique / number of samples) < `max_categorical_ratio`

Creating a Dataset
======================

From a DataFrame
~~~~~~~~~~~~~~~~
The default dataset constructor is expecting to get a dataframe. the rest of the properties
are optional, but if your data have date/index/label you would want to define them.

.. code-block:: python

    Dataset(my_dataframe, features=['feat1', 'feat2', 'feat3'], label='target', index='id', date='timestamp')

From Numpy Arrays
~~~~~~~~~~~~~~~~~~~
A Dataset can be created using a 2D numpy array for features and 1D numpy array for the labels. The features array is mandatory, and the labels array is optional.

.. code-block:: python

    features = np.array([[0.25, 0.3, 0.3], [0.14, 0.75, 0.3], [0.23, 0.39, 0.1]])
    labels = np.array([0.1, 0.1, 0.7])
    dataset_with_labels = Dataset.from_numpy(features, labels)
    dataset_without_labels = Dataset.from_numpy(features)

Also, it's possible to assign names to the features and label:

.. code-block:: python

    Dataset.from_numpy(
        features, labels,
        feature_names=['feat1', 'feat2', 'feat3',],
        label_name='target'
    )

All the rest of the Dataset's properties can be passed also as a regular keyword arguments:

.. code-block:: python

    Dataset.from_numpy(
        features, labels,
        feature_names=['feat1', 'feat2', 'feat3',],
        label_name='target',
        max_float_categories=10
    )
