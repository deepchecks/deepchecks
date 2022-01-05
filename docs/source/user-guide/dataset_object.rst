.. _dataset_object:

====================
The Dataset Object
====================

A Quick Introduction
====================

The Dataset Object is a simple and robust Object to store tabular data and metadata relevant to Machine Learning.

Common information used in Machine Learning that is aggregated in the Dataset:
 - features
    Sometimes called `X`, this is the features that are passed to the model.
 - categorical features
    A subset of the features. These features normally require some preprocessing before being passed to the model.
 - label
    Sometimes called `y`, the classes of the classification problem or values of a regression problem.
 - index
    The index of the data samples.
 - date
    A column containing the date the sample was acquired (Useful for TimeSeries problems).

All these and more can be defined and accessed on the Dataset Object.
This allows the user the flexibility to define these column names whichever they seam fit,
and allows use of the these tags to standardize development around this data.


The Basics
==========

Lets start with building a simple pd.DataFrame that we might already know as a popular data structure to store data.
But we also have other metadata that we have other use for, and not only as features for the model.

.. code-block:: python

    d = {"id": [1,2,3,4],
         "feature1": [0.1,0.3,0.2,0.6],
         "feature2": [4,5,6,7],
         "cat_feature": [0,0,0,1],
         "class": [1,2,1,2]}
    df = pd.DataFrame(d)

    dataset = Dataset(df, label="class", index_name="id", cat_features=["category"])

Now we have built a Dataset that we can easily pull features and labels from:

.. code-block:: python

    dataset.features

output:

::

    ['feature1', 'feature2', 'category']

.. code-block:: python

    dataset.label_name

output:

::

    ['class']

We can also get pd.DataFrames from columns that interest us e.g.:

.. code-block:: python

    dataset.features_columns

output:

::

        feature1	feature2	category
    0	0.1	        4	        0
    1	0.3	        5	        0
    2	0.2	        6	        0
    3	0.6	        7	        1

Now we can see that only features columns are pulled, which is save us the hassle
of tracking externally what is a feature column and what isn't one.

We can see the same thing happening here with the label_column. Instead of tracking an X DataFrame and a y DataSeries,
we have it all in one single object:

.. code-block:: python

    dataset.label_col

output:

::

        class
    0	1
    1	2
    2	1
    3	2



now instead of needing keeping track fo the label column, we can track it within the the Dataset Object.


 - Explain Cat Features and that we will talk about them later
 - Show some functionality that is intuitive and saves time (e.g. `train_test_split`)


Inferring Features And Categorical Features
===========================================
Dataset defines which columns of the data are features and of them which are categorical features.
For features, if parameter `features` not passed explicitly, all will be considered features apart from the label, index and date.
For categorical features, if parameter `cat_features` not passed explicitly, the following logic runs on every column to determine
whether the column is categorical or not:

* if columns is float type:
    * number of unique < `max_float_categories`
* else:
    *  number of unique < `max_categories` AND (number of unique / number of samples) < `max_categorical_ratio`

Useful Functions
================

Create Dataset From a Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


Train Test Split
~~~~~~~~~~~~~~~~

The same Function we all know and love, but saves a step:

.. code-block:: python

    train_dataset, test_dataset = dataset.train_test_split()


Link To API Referance
=====================


===============
Old Dataset Doc
===============


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
      - pandas.Series or Hashable
      - Data of labels as separate series from the data or name of label column in the data
      - None
    * - features
      - List[Hashable]
      - Names of the features in the data
      - None
    * - cat_features
      - List[Hashable]
      - Names of the categorical features in the data. Must be subset of `features`
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