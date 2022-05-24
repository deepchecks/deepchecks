============================
H2O
============================

`H2O <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html>`__ is
a widely used AutoML framework, written in Java but with multiple fully
implemented API libraries, including one in python. It it's ease of use and powerful results make it popular for
benchmarking and production-grade use alike.

Check out our interactive `example notebook <https://colab.research.google.com/github/deepchecks/deepchecks/blob/main/examples/integrations/h2o/deepchecks_h2o_tutorial.ipynb>`__,
demonstrating how to effortlessly test your H2O model with deepchecks in the exact same way
you can test any sklearn model! The notebook itself is available on Github `here <https://github.com/deepchecks/deepchecks/blob/main/examples/integrations/h2o/deepchecks_h2o_tutorial.ipynb>`__.

**The example notebook includes:**

1. Training a simple H2O model.
2. Demonstrating how use deepchecks on the H2O model to run a check. We'll run a deepchecks model performance check , that generates a detailed plot for model comparison (we'll compare the H2O model's performance to a a scikit-learn based model).
3. Using deepchecks for gaining a deeper understanding about the difference in performance between the two models.


Quick peek at the code for running the check on an H2O-based model:

.. code:: ipython3

    from deepchecks.tabular.integrations.h2o import H2OWrapper
    from deepchecks.tabular.checks import MultiModelPerformanceReport

    h2o_deepchecks_model = H2OWrapper(h2o_model)
    MultiModelPerformanceReport().run(train_dataset, test_dataset, [h2o_deepchecks_model, sklearn_model])
