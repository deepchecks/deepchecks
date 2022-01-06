.. _dataset_object:

====================
The Dataset Object
====================


A Quick Introduction
====================

The Dataset Object is a simple and robust Object to store tabular data and metadata relevant for Machine Learning.
It allows deepchecks to understand the context of the data with minimal hassle for the user.

The Dataset's common metadata properties:
 - **label** - Represent the value we are trying to predict in the data.
 - **cat_features** - Categorical features normally require some preprocessing before being passed to the model. If not specified, the categorical features are inferred automatically from the data itself. (`Inferring Features And Categorical Features`_ )
 - **index_name** - Some datasets have unique index that may be meaningful. Like the other arguments, this is not mandatory.
 - **date_name** - A date column representing the sample. (Useful for TimeSeries problems).
 - **features** - Specifies the columns used by model for training. Used for defining only a subset of the columns in the data as features. If not defined, all columns that aren't explicitly specified as `label`, `date`, or `index` are considered as features.

All these and more can be defined and accessed on the :doc:`Dataset Object <api/base/generated/deepchecks.base.dataset.Dataset>`.
This allows the user the flexibility to define these column names whichever they seem fit,
and allows use of the these tags to standardize development around this data.


The Basics
==========

Lets start with building a simple `pd.DataFrame` that we might already know as a popular data structure to store data, and demonstrate how the Dataset object should be defined.

>>> d = {"id": [1,2,3,4],
...      "feature1": [0.1,0.3,0.2,0.6],
...      "feature2": [4,5,6,7],
...      "categorical_feature": [0,0,0,1],
...      "class": [1,2,1,2]}
... df = pd.DataFrame(d)
... dataset = Dataset(df, label="class", index_name="id", cat_features=["categorical_feature"])

Now that we have a Dataset we can also inspect its defined features and label:

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

Now we can see that only features columns are pulled, which save us the hassle
of tracking externally what is a feature column and what isn't one.

We can see the same thing happening here with the label_column. Instead of tracking an X DataFrame and a y DataSeries,
we have it all in one single object:

    >>> dataset.label_col
        class
    0	1
    1	2
    2	1
    3	2


Now instead of needing keeping track fo the label column, we can track it within the the Dataset Object.

Inferring Features And Categorical Features
===========================================
The Dataset defines which columns of the data are considered as the model's features, and of them, which are treated as categorical features.
For features, if parameter `features` not passed explicitly, all columns besides the the `label`, `index `and `date `will be considered as features.
For categorical features, if the parameter `cat_features` was not passed explicitly, the following inference logic will run on each column to determine
how it will be treated:

* if columns is float type:
    * number of unique < `max_float_categories`
* else:
    *  number of unique < `max_categories` AND (number of unique / number of samples) < `max_categorical_ratio`

Useful Functions
================

Create a Dataset from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A Dataset can be created using a 2D numpy array for features and 1D numpy array for the labels. The features array is mandatory, and the labels array is optional.

>>> features = np.array([[0.25, 0.3, 0.3], [0.14, 0.75, 0.3], [0.23, 0.39, 0.1]])
>>> labels = np.array([0.1, 0.1, 0.7])
>>> dataset_with_labels = Dataset.from_numpy(features, labels)
>>> dataset_without_labels = Dataset.from_numpy(features)

Also, it's possible to assign names to the features and label:

>>> Dataset.from_numpy(
...     features, labels,
...     columns=['feat1', 'feat2', 'feat3'],
...     label_name='target'
... )

All the rest of the Dataset's properties can be passed also as regular keyword arguments:

>>> Dataset.from_numpy(
...     features, labels,
...     columns=['feat1', 'feat2', 'feat3'],
...     label_name='target',
...     max_float_categories=10
... )


Train Test Split
~~~~~~~~~~~~~~~~

Uses internally `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_ (so same arguments can be passed) and also copies the metadata to each instance of the split and returns two Datasets.

>>> train_dataset, test_dataset = dataset.train_test_split(stratify=True)


Copy
~~~~

Copy allows copying of metadata on a new dataframe. This can be used on new data to help compare datasets.

>>> new_dataset = dataset.copy(new_data_df)


Dataset API Reference
==================

.. currentmodule:: deepchecks.base.dataset

.. autosummary::

    Dataset
