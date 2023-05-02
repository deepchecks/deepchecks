.. _nlp__metadata_guide:

=================
NLP Metadata
=================

Metadata is any additional information about the texts, that can be used to better understand the data and the model's
behavior. In contrast to unstructured text, metadata is structured, and is expected to be a pandas DataFrame.
Metadata can be anything - from the source of the data, the timestamp of its creation, to the age of the author, and
any other information that is given about the text.

Not to be confused with :ref:`text properties <nlp__properties_guide>`, which are features extracted from the text
itself, such as text length or sentiment.

What Is Metadata Used For?
=============================

Metadata is used by Deepchecks' checks to examine the dataset, without directly using the text, as some tests are
difficult to perform on unstructured data.
Inspecting the distribution of a metadata column's values (e.g. to notice that the source of the texts is different
between the train and test sets) can help uncover potential problems in the way that the datasets were built,
or hint about the model's expected performance on unseen data.

Example for specific scenarios in which measuring the metadata may come in handy:

#. **Find weak segments** - The metadata can be used to segment the data and test for low performing segments.
   If found, the weak segment may indicate an underrepresented segment or an area where the data quality is worse.
#. **Find obscure relations between the data and the targets** - the model training might be affected
   by the metadata, even if it doesn't have access to it.
   For example, in a classification dataset of toxic statements, if the toxic statements came from a specific source
   (e.g. Twitter), the model might learn to predict the label by other traits of the source (e.g. the allowed length of
   the text). In this case, the model will perform well on the training data, and may even perform well on the test
   data, but will fail to generalize to new data.


How To Use Metadata
=====================

Metadata Format
---------------
Metadata must be given as a pandas DataFrame, with the rows representing each sample and columns representing the
different metadata columns. The number of rows in the metadata DataFrame must be equal to the number of samples in the
dataset, and the order of the rows must be the same as the order of the samples in the dataset.

How To Pass Metadata To Deepchecks
-----------------------------------
In order to pass metadata to Deepchecks, you must pass it to the :class:`TextData <deepchecks.nlp.TextData>` object.
The metadata can be passed to the :class:`TextData <deepchecks.nlp.TextData>` object in two ways:

#. When initializing the :class:`TextData <deepchecks.nlp.TextData>` object, by passing the metadata to the
   ``metadata`` parameter.
#. After initializing the :class:`TextData <deepchecks.nlp.TextData>` object, by using the ``TextData.set_metadata``
   function.

In both methods, you can pass the metadata as a pandas DataFrame, or as a path to a csv file.

Additionally, it's advised to also use the ``categorical_metadata`` parameter to specify which metadata columns are
categorical. The parameter should be a list of the names of the categorical columns.


In the following example, we will pass the metadata to the ``TextData`` object in order to use the
``MetadataSegmentsPerformance`` check:

.. code-block:: python

    from deepchecks.nlp import TextData
    from deepchecks.nlp.checks import MetadataSegmentsPerformance

    # Load the data
    train_texts, train_labels = load_train_data()
    test_texts, test_labels = load_test_data()

    # Load the metadata
    train_metadata = load_train_metadata()
    test_metadata = load_test_metadata()

    # Option 1: Initialize the TextData object with the metadata
    train = TextData(train_texts, train_labels, metadata=train_metadata, categorical_metadata=['source'])

    # Option 2: Initialize the TextData object without the metadata, and set it later
    test = TextData(test_texts, test_labels)
    test.set_metadata(test_metadata, categorical_metadata=['source'])

    # Run the check
    check = MetadataSegmentsPerformance()
    check.run(train, test)
