.. _dataset_object:

====================
The Dataset Object
====================


A Quick Introduction
====================

The Dataset Object is a simple and robust Object to store tabular data and metadata relevant to Machine Learning.

Common information used in Machine Learning that is aggregated in the Dataset:
 - features
    List of column names. This parameter specifies the features the model is trained on.
    If not defined, columns that are not defined as label, date, or index are considered as features.
 - cat_features
    List of column names. A subset of the features. Categorical features normally require some preprocessing before being passed to the model.
    If not specified, the categorical features are inferred automatically from the data itself. (More info in :ref:`Inferring Features And Categorical Features`)
 - label
    Either name of a column, or `pd.Series`. The classes of the classification problem or values of a regression problem.
 - index_name
    Name of the index column. This can be useful to track indexes duplicate etc. Like the other arguments, this is not manditory.
 - date_name
    Name of the date column. The date the sample was acquired (Useful for TimeSeries problems). This column is standardized using `pd.Timestamp`.

All these and more can be defined and accessed on the Dataset Object.
This allows the user the flexibility to define these column names whichever they seam fit,
and allows use of the these tags to standardize development around this data.


The Basics
==========

Lets start with building a simple pd.DataFrame that we might already know as a popular data structure to store data.
But we also have other metadata that we have other use for, and not only as features for the model.

    >>> d = {"id": [1,2,3,4],
    ...      "feature1": [0.1,0.3,0.2,0.6],
    ...      "feature2": [4,5,6,7],
    ...      "cat_feature": [0,0,0,1],
    ...      "class": [1,2,1,2]}
    ... df = pd.DataFrame(d)
    ... dataset = Dataset(df, label="class", index_name="id", cat_features=["category"])

Now we have built a Dataset that we can easily pull features and labels from:

    >>> dataset.features
    ['feature1', 'feature2', 'category']
    >>> dataset.label_name
    ['class']

We can also get pd.DataFrames from columns that interest us e.g.:

    >>> dataset.features_columns
        feature1	feature2	category
    0	0.1	        4	        0
    1	0.3	        5	        0
    2	0.2	        6	        0
    3	0.6	        7	        1

Now we can see that only features columns are pulled, which is save us the hassle
of tracking externally what is a feature column and what isn't one.

We can see the same thing happening here with the label_column. Instead of tracking an X DataFrame and a y DataSeries,
we have it all in one single object:

    >>> dataset.label_col
        class
    0	1
    1	2
    2	1
    3	2


now instead of needing keeping track fo the label column, we can track it within the the Dataset Object.

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

    >>> features = np.array([[0.25, 0.3, 0.3], [0.14, 0.75, 0.3], [0.23, 0.39, 0.1]])
    >>> labels = np.array([0.1, 0.1, 0.7])
    >>> dataset_with_labels = Dataset.from_numpy(features, labels)
    >>> dataset_without_labels = Dataset.from_numpy(features)

Also, it's possible to assign names to the features and label:

    >>> Dataset.from_numpy(
    ...     features, labels,
    ...     feature_names=['feat1', 'feat2', 'feat3',],
    ...     label_name='target'
    ... )

All the rest of the Dataset's properties can be passed also as a regular keyword arguments:

    >>> Dataset.from_numpy(
    ...     features, labels,
    ...     feature_names=['feat1', 'feat2', 'feat3',],
    ...     label_name='target',
    ...     max_float_categories=10
    ... )


Train Test Split
~~~~~~~~~~~~~~~~

Using the `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_ but also copies the metadata to each instance of the split.

    >>> train_dataset, test_dataset = dataset.train_test_split()


Copy
~~~~

Copy allows copying of metadata on a new dataframe. This can be used on data that already is passed the train test split.

    >>> test_dataset = train_dataset.copy(test_df)


Full API Reference
==================

.. currentmodule:: deepchecks.base.dataset

.. autosummary::

    Dataset
