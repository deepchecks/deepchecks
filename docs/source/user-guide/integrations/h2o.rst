============================
H2O
============================

`H2O <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html>`__ is
a widely used AutoML framework, written in Java but with multiple fully
implemented API libraries, including one in python. It it's ease of use and powerful results make it popular for
benchmarking and production-grade use alike.

Check out our interactive `example notebook <https://colab.research.google.com/github/deepchecks/deepchecks/tree/examples/integrations/h2o/deepchecks_h2o_tutorial.ipynb>`__,
demonstrating how to effortlessly test your H2O model with deepchecks in the exact same way
you can test any sklearn model!

In this example includes:
-------------------------

1. Training a simpel H2O model.
2. How to use deepchecks to get beautiful plots comparing it's performance to an sklearn model.
3. Using deepchecks to understand the deeper differences between the models.


In short:

.. code:: ipython3

    from deepchecks.tabular.integrations.h2o import H2OWrapper
    from deepchecks.tabular.checks import MultiModelPerformanceReport

    h2o_deepchecks_model = H2OWrapper(h2o_model)
    MultiModelPerformanceReport().run(train_dataset, test_dataset, [h2o_deepchecks_model, sklearn_model])
