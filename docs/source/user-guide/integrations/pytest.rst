======
Pytest
======

This tutorial demonstrates how deepchecks can be used inside unit tests performed on data or model, with the pytest
framework.
We will use the ``diabetes`` dataset from scikit-learn, and check whether certain columns contain drift
between the training and the test sets.

.. code-block:: python

    import pytest
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    from deepchecks import Dataset
    from deepchecks.tabular.checks import TrainTestFeatureDrift
    from deepchecks.tabular.suites import data_integrity

Defining Pytest Fixtures
-------------------------

pytest fixtures provide a defined, reliable and consistent context for the tests. This could include environment (for
example a database configured with known parameters) or content (such as a dataset).
In this tutorial we will define a fixture that load the ``diabetes`` dataset from scikit-learn.

.. code-block:: python

    @pytest.fixture(scope='session')
    def diabetes_df():
        diabetes = load_diabetes(return_X_y=False, as_frame=True).frame
        return diabetes

Implementing the Test
-----------------------

Now, we will implement a test that will check if some columns in the dataset have drifted between the train and test datasets.
the test sets.

.. code-block:: python

    def test_diabetes_drift(diabetes_df):
        train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
        train = Dataset(train_df, label='target', cat_features=['sex'])
        test = Dataset(test_df, label='target', cat_features=['sex'])

        check = TrainTestFeatureDrift(columns=['age', 'sex', 'bmi'])
        check.add_condition_drift_score_not_greater_than(max_allowed_psi_score=0.2,
                                                         max_allowed_earth_movers_score=0.1)

        result = check.run(train, test)

        assert result.passed_conditions()

Please note the :meth:`passed_conditions() <deepchecks.core.CheckResult.passed_conditions>` method of the :class:`deepchecks.core.CheckResult` object. This method will return ``True`` if all the
conditions are met, and ``False`` otherwise.

It's possible to evaluate the result of a suite of checks, and to get the overall result of the test, by using the
:meth:`deepchecks.core.SuiteResult.passed` method.

.. code-block:: python

    def test_diabetes_integrity(diabetes_df):
        ds = Dataset(diabetes_df, label='target', cat_features=['sex'])

        suite = data_integrity()
        result = suite.run(ds)

        assert result.passed(fail_if_warning=True, fail_if_check_not_run=False)