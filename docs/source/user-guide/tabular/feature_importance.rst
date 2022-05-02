.. _dataset_object:

====================
The Dataset Object
====================

#TODO:
link to shap
link to unused features
link to feature drift
sklearn.inspection.permutation_importance
link to "how to make faster"


What is Feature Importance?
==================
Feature importance is a ranking that represents the significance of the input features to the model's predictions.
A feature with higher importance has more influence on the prediction of the model.
There are many ways to score the feature importance, some are general (such as Shapley values) and some are specific
for the specific model



Why Does Deepchecks Use Feature Importance?
==================
Deepchecks uses your model's feature importance for 2 main reasons:
* Help you find issues with your model or data, as in the check UnusedFeatures
* Display the most relevant information the check has found (for instance, if deepchecks found drift in many features,
as in the check FeatureDrift it would only display the features with the highest feature importance that have the most
effect on your model)

Note
-----
Please note that for most checks, deepchecks does not *require* the usage of feature importance, and that you can
shorten or even skip this phase of the calculation. See "... "

How Do I Insert My Feature Importance So Deepchecks May Use It?
==================
The easiest way to save time in the calculation of feature importance is to supply deepchecks with your own data on
feature importance.
This can be done by using the ``feature_importance`` input received by the ``run`` function, available in all
checks and suites.

What If I Don't Have My Feature Importance?
-----
In this case, deepchecks will calculate the feature importance for you.
You can also force this action even if your model has built-in feature importance by using the
``feature_importance_force_permutation`` input received by the ``run`` function, available in all checks and suites.


How Does Deepchecks Calculate Feature Importance?
==================
First, deepchecks makes sure

, or by enabling your model calculate its own feature importance according to the common Scikit-Learn
format (meaning, your model needs to have the attribute ``feature_importances_`` or ``coef_``

permutation_importance



If you don't have your model's feature importance and want deepchecks to skip the calculation of feature importance
anyway, you can just change the ``feature_importance_timeout`` input to 0.




The ``Dataset`` is a container for the data and the relevant ML metadata such as special column roles (e.g. label, index, categorical columns).
It enables to take into account the relevant context during validation,
and to save it in a convenient manner, and is a basic building block in deepchecks.


Class Properties
==================

The common properties are:

- **label** - The target values that the model is trying to predict.
- **cat_features** - List of features that should be treated as categorical. If not specified explicitly, they will be :ref:`inferred automatically <dataset_object__inferring_categorical_features>`.
- **index_name** - If the dataset has a meaningful unique index, defining it as such will enable more validations to run.
- **date_name** - A date column representing the sample.
- **features** - Specifies the columns used by model for training.
  Used for defining only a subset of the columns in the data as features. If not supplied then
  all of the columns that aren't explicitly specified as ``label``, ``date``, or ``index`` are considered to be features.

The Dataset's metadata properties are all optional. Check out the API Reference for more details.

Dataset API Reference
--------------------------

.. currentmodule:: deepchecks.tabular.dataset

.. autosummary::

    Dataset


Creating a Dataset
=======================

From a Pandas DataFrame
--------------------------

The default ``Dataset`` constructor expects to get a ``pd.DataFrame``
The rest of the properties are optional, but if your data has ``date``/``index``/``label``
columns you would want to define them for more possible validation checks.

>>> d = {"id": [1,2,3,4],
...      "feature1": [0.1,0.3,0.2,0.6],
...      "feature2": [4,5,6,7],
...      "categorical_feature": [0,0,0,1],
...      "class": [1,2,1,2]}
... df = pd.DataFrame(d)
... ds = Dataset(df, label="class", index_name="id", cat_features=["categorical_feature"])


From Numpy Arrays
---------------------

A Dataset can be created using a 2D numpy array for features and 1D numpy array for the labels.
The features array is mandatory, and the labels array is optional.

>>> features = np.array([[0.25, 0.3, 0.3], [0.14, 0.75, 0.3], [0.23, 0.39, 0.1]])
>>> labels = np.array([0.1, 0.1, 0.7])
>>> ds_with_labels = Dataset.from_numpy(features, labels)
>>> ds_without_labels = Dataset.from_numpy(features)

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

Useful Functions
===================

Train Test Split
--------------------

Uses internally `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_
(so the same arguments can be passed) and also copies the metadata to each instance of the split and returns two ``Datasets``.

>>> train_ds, test_ds = ds.train_test_split(stratify=True)


Copy
------

Copy enables to copy the metadata from an existing ``Dataset`` instance, for creating a new ``Dataset`` from a new ``pd.DataFrame``'s data.
This can be useful for implementing data splits independentaly or for comparing datasets, when receiving new data (of the same known format of existing data).

>>> new_ds = ds.copy(new_df)


Working with Class Parameters
---------------------------------

We can work directly with the ``Dataset`` object, to inspect its defined features and label:

>>> ds.features
['feature1', 'feature2', 'category']
>>> ds.label_name
['class']

Get its internal ``pd.DataFrame``:

>>> ds.data
    feature1    feature2    categorical_feature class    
0   0.1         4           0                   1
1   0.3         5           0                   2
2   0.2         6           0                   1
3   0.6         7           1                   2


Or extract directly only the feature columns or only the label column from within it:

>>> ds.features_columns
    feature1	feature2	categorical_feature
0	0.1	        4	        0
1	0.3	        5	        0
2	0.2	        6	        0
3	0.6	        7	        1

>>> ds.label_col
    class
0   1
1   2
2   1
3   2


.. _dataset_object__inferring_categorical_features:

Inferring Categorical Features
==================================

.. warning::
    It is highly recommended to explicitly state the categorical features or define their column type to be ``category``.
    Otherwise, the inherent limitations of the automatic, and may cause inconsistencies (misdetection, different detection between
    train and test, etc.), and required tuning and adaptions.

If the parameter ``cat_features`` was not passed explicitly, the following inference logic 
will run on the columns to determine which are classified as categorical:

#. If the ``pd.dtypes`` of any of the existing columns is ``category`` then all of the columns that are of type ``category`` 
   will be  considered categorical (and only them).

#. Otherwise, a heuristic is used for deducting the type. Each column for which at least one of the following conditions is met is considered categorical:

   - If (`number of unique values in column` <= `max_float_categories`) 
     **AND** (`column type` is `float`)

   - If (`number of unique values in column` <= `max_categories`)  
     **AND** ((the ratio between the `number of unique values` and the `number of samples`) < `max_categorical_ratio`)
  
Check the API Reference for :doc:`infer_categorical_features </api/generated/deepchecks.utils.features.infer_categorical_features>`
for more details.