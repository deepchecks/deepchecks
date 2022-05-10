============================
H2O
============================

`H2O <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html>`__ is
a widely used AutoML framework, written in Java but with multiple fully
implemented API libraries, including one in python. In this example,
we’ll be showing how you can effortlessly test your H2O model with
deepchecks in the exact same way you can test any sklearn model.

This example can be run using
`collab <https://colab.research.google.com/github/deepchecks/deepchecks/tree/examples/integrations/h2o/deepchecks_h2o_tutorial.ipynb>`__.

Building an H2O Model
---------------------

First we’ll load our dataset, using the adult dataset that can be
downloaded from the `UCI machine learning
repository <http://archive.ics.uci.edu/ml>`__ and is also available
in `deepchecks.tabular.datasets`.

.. code:: ipython3

    import warnings
    warnings.filterwarnings('ignore')
    from deepchecks.tabular.datasets.classification import adult

.. code:: ipython3

    train_df, test_df = adult.load_data(data_format='Dataframe', as_train_test=True)

Now we’ll build an H2O `H2OGradientBoostingEstimator <https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2o.estimators.gbm.H2OGradientBoostingEstimator>`__ based on this
dataset split.

.. code:: ipython3

    import h2o
    h2o.no_progress()
    h2o.init()


H2O models can only train and predict on H2OFrame objects.

.. code:: ipython3

    train_h2o = h2o.H2OFrame(train_df)

.. code:: ipython3

    gbm = h2o.estimators.gbm.H2OGradientBoostingEstimator();

We must tell the H2O model what are the feature columns and what is the
name of the label column.

.. code:: ipython3

    feature_cols = adult._FEATURES
    print(feature_cols)
    label_col = adult._target
    print(label_col)


.. code:: ipython3

    gbm.train(x=feature_cols, y=label_col, training_frame=train_h2o, verbose=False);


To compare to our new H2O model, we’ll download the default scikit-learn
model (a pipeline containing a naive Random Forest Classifier) for this
task from deepchecks.

.. code:: ipython3

    skmodel = adult.load_fitted_model()

Running deepchecks on H2O
=========================

before we can run deepchecks on our H2O model, we have one step
remaining - bridging between the API of the H2O model and the
scikit-learn API that deepchecks expects.

We will write a wrapper to our H2O model, that will implement the required API for deepchecks according the the
:doc: `</user-guide/tabular/supported_models>` guide. This wrapper is also available for import under
`deepchecks.tabular.integrations.h2o` but is brought here for completes.

Generally the wrapper model will contain 2 functions in
case of a classification problem: the ``predict`` and the ``predict_proba`` functions that will be called by
deepchecks. In addition it is also possible to specify the feature importances of the model. Read more about
feature importance handling in the :doc: `</user-guide/tabular/feature_importance>` guide.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    class H2OWrapper:
    
        def __init__(self, h2o_model):
            self.model = h2o_model
            
        def predict(self, df: pd.DataFrame) -> np.array:
            return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 0]
    
        def predict_proba(self, df: pd.DataFrame) -> np.array:
            return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 1:].astype(float)

.. code:: ipython3

    h2o_dc_gbm = H2OWrapper(gbm)

Now we can run deepchecks! We’ll start by defining the deepchecks
Dataset objects and comparing the two models.

.. code:: ipython3

    from deepchecks.tabular import Dataset

.. code:: ipython3

    train_ds = Dataset(train_df, label=label_col, features=feature_cols, cat_features=adult._CAT_FEATURES)
    test_ds = Dataset(test_df, label=label_col, features=feature_cols, cat_features=adult._CAT_FEATURES)

Comparing model performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from deepchecks.tabular.checks import MultiModelPerformanceReport

.. code:: ipython3

    MultiModelPerformanceReport().run(train_ds, test_ds, [h2o_dc_gbm, skmodel])


We can see that from an overall performance standpoint (considering the
f1 metric), the H2O model achieves superior performance compared to the
sklearn model, especially in the smaller class.

Further model performance evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We next wish to get a more in depth evaluation of the H2O model. For example,
we can use the SegmentPerformance check to compare the performance of
the two models across segments of the sex and relationship status
features.

.. code:: ipython3

    from deepchecks.tabular.checks import SegmentPerformance
    # For a reason that has to do with the way h2o handles multiprocessing, we must change the parallel_backend from
    # loky to threading for all the of the checks to be able to compute while the h2o model is inferring.
    from sklearn.utils import parallel_backend
    with parallel_backend('threading', n_jobs=2):
        result_h2o = SegmentPerformance(feature_1='sex', feature_2='relationship').run(test_ds, h2o_dc_gbm)
        
    result_skl = SegmentPerformance(feature_1='sex', feature_2='relationship').run(test_ds, skmodel)


.. parsed-literal::

    Calculating permutation feature importance. Expected to finish in 56 seconds


.. code:: ipython3

    result_h2o


.. code:: ipython3

    result_skl


We see that the scikit-learn model, apart from being worse overall, is
specifically worse (a drop in accuracy of 0.6) at predicting the income
of females with a relationship status of “Others”.
