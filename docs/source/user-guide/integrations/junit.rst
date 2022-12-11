======
JUnit
======

This tutorial demonstrates how deepchecks can be used to output junit from tests performed on data or model. This can
then be read in by junit parsers within CI/CD pipelines. This allows a more generic, but less adaptable, integration
compared to the pytest integration where a developer needs to wrap every test manually. We will use the ``iris``
dataset from scikit-learn, and check whether certain columns contain drift between the training and the test sets.

General Structure
-------------------------
JUnit comprises of 3 sections:
1. The test suites sections. This is an optional section, but used with the Deepchecks Junit Serializer since multiple
suites are used.
2. The test suite section. This is used to group tests by their domain such as model, data, or evaluation test. A catch
all 'checks' section is also added for custom checks added.
3. The test case section. This is the atomic unit of the payload and contains the details about a
:class:`deepchecks.core.CheckResult`. This can either be a pass, failure, or skip.

It will output the following formatted string by default, but a XML can also be extracted:

<testsuites errors="0" failures="9" name="Full Suite" tests="54" time="45">
    <testsuite errors="0" failures="3" name="train_test_validation" tests="12" time="1" timestamp="2022-11-22T05:49:01">
        <testcase classname="deepchecks.tabular.checks.train_test_validation.feature_label_correlation_change.FeatureLabelCorrelationChange" name="Feature Label Correlation Change" time="0">
            <system-out>Passed for 4 relevant columns, Found 2 out of 4 features in train dataset with PPS above threshold: {'petal width (cm)': '0.91', 'petal length (cm)': '0.86'}</system-out>
        </testcase>
        ...
    </testsuite>
<testsuites>

Failed test cases will return in the following manner:

<testcase classname="deepchecks.tabular.checks.train_test_validation.date_train_test_leakage_duplicates.DateTrainTestLeakageDuplicates" name="Date Train Test Leakage Duplicates" time="0">
    <failure message="Dataset does not contain a datetime" type="failure">Check if test dates are present in train data.</failure>
</testcase>

Failed test cases can also be coerced to skipped to allow the recording of the failed test, but to not put a break into
a CI/CD system. This is especially helpful when first starting out with deepchecks in CI/CD pipelines.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    from deepchecks import Dataset
    from deepchecks.tabular.suites import full_suite
    from deepchecks.core.serialization.suite_result.junit import SuiteResultSerializer as JunitSerializer

Executing a Test Suite
-------------------------
.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from deepchecks.tabular import Dataset
    from sklearn.datasets import load_iris

    label_col = 'target'

    X, y = load_iris(return_X_y=True, as_frame=True)
    X[label_col] = y
    df_train, df_test = train_test_split(X, stratify=X[label_col], random_state=0)


    # Train Model
    rf_clf = RandomForestClassifier(random_state=0, max_depth=2)
    rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col])
    ds_train = Dataset(df_train, label=label_col, cat_features=[])
    ds_test = Dataset(df_test, label=label_col, cat_features=[])

    suite = full_suite()

    results = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

    output = JunitSerializer(results).serialize()
