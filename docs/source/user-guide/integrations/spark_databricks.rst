Spark & Databricks
==================

This tutorial demonstrates how deepchecks can be used on the Databricks ML platform using Spark. We will build a
logistic regression model on top of the Adult dataset, a sample dataset that is automatically available on every
databricks workspace.

Loading the dataset
-------------------
We first define the dataset schema and then load it as a Spark dataframe.

.. code-block:: python

    schema = """`age` DOUBLE,
    `workclass` STRING,
    `fnlwgt` DOUBLE,
    `education` STRING,
    `education_num` DOUBLE,
    `marital_status` STRING,
    `occupation` STRING,
    `relationship` STRING,
    `race` STRING,
    `sex` STRING,
    `capital_gain` DOUBLE,
    `capital_loss` DOUBLE,
    `hours_per_week` DOUBLE,
    `native_country` STRING,
    `income` STRING"""

    dataset = spark.read.csv("/databricks-datasets/adult/adult.data", schema=schema)

    # Splitting the data to train/test set
    trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)

Defining deepchecks Dataset
---------------------------
We first convert the spark DataFrame to a pandas dataframe deepchecks can work with.

.. note::
    Conversion to a pandas dataframe will load the data into memory. If you have a large dataset, is is recommended to
    sample the data first. Logically, it is OK to sample since anyway most of the checks will be performed on a small
    subset of the data.

.. code-block:: python

    from deepchecks.tabular import Dataset

    pd_train = trainDF.toPandas()
    pd_test = testDF.toPandas()

    ds_train = Dataset(pd_train, label='income', cat_features=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])
    ds_test = Dataset(pd_test, label='income', cat_features=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])

Running the Integrity Suite
---------------------------
One of deepchecks' use-cases is to validate the integrity of the dataset, even without a model. In order to do so, the
single dataset integrity suite can be run on the dataset.

.. code-block:: python

    from deepchecks.tabular.suites import data_integrity
    # Validate the training set
    train_res = data_integrity().run(ds_train)

Displaying the results
~~~~~~~~~~~~~~~~~~~~~~
We will use the built-in functions of the Databricks platform to view the results in a HTML format.

.. code-block:: python

    from io import StringIO
    buff = StringIO()
    train_res.save_as_html(buff)

    displayHTML(buff.getvalue())

Building a Logistic Regression Model
------------------------------------
After we validated that our data is clean and ready to be used in a model, we will build a logistic regression model
that classifies whether a person's income is above or below 50K.
First, we preprocess the features. The categorical features are one-hot encoded and the label is being transformed to
0/1.

.. code-block:: python

    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

    categoricalCols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

    # The following two lines are estimators. They return functions that we will later apply to transform the dataset.
    stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols])
    encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols])

    # The label column ("income") is also a string value - it has two possible values, "<=50K" and ">50K".
    # Convert it to a numeric value using StringIndexer.
    labelToIndex = StringIndexer(inputCol="income", outputCol="label")

    stringIndexerModel = stringIndexer.fit(trainDF)

    # This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
    numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
    vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

Training the Model
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline

    lr = LogisticRegression(featuresCol="features", labelCol="label")

    # Define the pipeline based on the stages created in previous steps.
    pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

    # Fit the pipeline model.
    pipelineModel = pipeline.fit(trainDF)

Writing a Model Wrapper
~~~~~~~~~~~~~~~~~~~~~~~
We will write a wrapper to our model, that will implement the required API for deepchecks according the the
:doc:`supported models </user-guide/tabular/supported_models>` guide. Generally the wrapper model will contain 2 functions in
case of a classification problem: the ``predict`` and the ``predict_proba`` functions that will be called by
deepchecks. In addition it is also possible to specify the feature importances of the model. Read more about
feature importance handling in the :doc:`feature importance </user-guide/tabular/feature_importance>` guide.

.. code-block:: python

    import numpy as np
    import pyspark
    from pyspark.ml.feature import IndexToString

    class PySparkModelWrapper:
        def __init__(self, model: pyspark.ml.pipeline.PipelineModel, label_map):
            self.model = model
            self.idx_to_string = IndexToString(inputCol="prediction", outputCol="predictedValue")
            self.idx_to_string.setLabels(label_map)

        def predict(self, X: np.ndarray) -> np.ndarray:
            df=spark.createDataFrame(X)
            preds = self.idx_to_string.transform(self.model.transform(df).select('prediction')).select('predictedValue').collect()
            return np.array(preds).reshape(-1)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            df=spark.createDataFrame(X)
            preds = self.model.transform(df).select('prediction').collect()
            return np.array(preds).reshape(-1, 2)

        @property
        def feature_importances_(self):
            return np.array([1/14] * 14)

.. note::
    The wrapper here considers that all features are equally important. This is not a valid assumption for
    real models, but is done here for simplicity.

Evaluating the Model Using Deepchecks Suites
--------------------------------------------

We will run 2 suites, the ``model_evaluation`` suite that is meant to test model performance and overfit, and the
``train_test_validation`` is meant to validate correctness of train-test split, including integrity, distribution and
leakage checks.

.. code-block:: python

    from deepchecks.tabular.suites import model_evaluation, train_test_validation

    eval_suite = model_evaluation()
    model_evaluation_res = eval_suite.run(ds_train,ds_test, PySparkModelWrapper(pipelineModel,
                                          pipelineModel.stages[2].labels))

    train_test_suite = train_test_validation()
    train_test_res = train_test_suite.run(ds_train, ds_test, PySparkModelWrapper(pipelineModel,
                                          pipelineModel.stages[2].labels))

Displaying the Results
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from io import StringIO
    buff = StringIO()
    model_evaluation_res.save_as_html(buff)

    displayHTML(buff.getvalue())

    buff = StringIO()
    train_test_res.save_as_html(buff)

    displayHTML(buff.getvalue())
