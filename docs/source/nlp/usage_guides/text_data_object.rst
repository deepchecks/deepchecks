.. _nlp__textdata_object:

===================
The TextData Object
===================

The :class:`TextData <deepchecks.nlp.TextData>` is a container for your textual data, labels, and relevant
metadata for NLP tasks and is a basic building block in the ``deepchecks.nlp`` subpackage.
In order to use any functionality of the ``deepchecks.nlp`` subpackage, you need to first create a ``TextData`` object.
The ``TextData`` object enables easy access to metadata, embeddings and properties relevant for training and validating ML
models.

Class Properties
==================

The main properties are:

- **raw_text** - The raw text data, a list of strings representing the raw text of each sample. Each sample can be a
  sentence, paragraph, or a document, depending on the task.
- **label** - The labels for the text data samples.
- **task_type** - The task type, must be either `text_classification`, `token_classification` or None. See the
  :ref:`Supported Tasks Guide <nlp__supported_tasks>` for more information about supported formats.

TextData API Reference
-------------------------

.. currentmodule:: deepchecks.nlp.text_data

.. autosummary::

    TextData


Creating a TextData
=======================

The default ``TextData`` constructor expects to get a sequence of raw text strings or tokenized text.
The rest of the arguments are optional, but if you have labels for your data you would want to define them in the constructor,
as many checks require the dataset labels in order to run.

.. admonition:: Defining task_type
   :class: attention

   If you define labels, you must also define the ``task_type`` so deepchecks will know how to parse the labels.


>>> raw_text = ["This is an example.", "Another example here."]
>>> labels = ["positive", "negative"]
>>> task_type = "text_classification"
>>> text_data = TextData(raw_text=raw_text, label=labels, task_type=task_type)

Tokenized Text
----------------

If you have tokenized text, you can also create a TextData object from it rather than using the ``raw_text`` argument:

>>> # A tokenized example with named entities and locations
>>> tokenized_text = [["Dan", "lives", "in", "New", "York", "."], ["He", "works", "at", "Google", "."]]
>>> labels = [["B-PER", "O", "O", "B-LOC", "I-LOC", "O"], ["O", "O", "O", "B-ORG", "O"]]
>>> text_data = TextData(tokenized_text=tokenized_text, label=labels, task_type=task_type)

If you're running deepchecks on a token classification task it is recommended to use that argument instead of the
``raw_text`` argument. If you did pass ``raw_text`` to the constructor,
deepchecks will break the text into tokens for you, using the default python ``str.split()`` method to split the text
into tokens.

Useful Functions
===================

Describe data
-----------------------------

To get an overall view of the data, you can use the `describe()` function that will display label distribution, some statistical
information regarding the data such as number of samples, annotation ratio, metadata, properties, etc. and will generate distribution
plots of the text properties. You can use the function on the TextData object using:

>>> text_data.describe()

Calculate Default Properties
-----------------------------

To calculate all the default properties, you do not need to pass the ``include_properties`` parameter in the
``calculate_builtin_properties`` function. If you pass either ``include_properties`` or ``ignore_properties`` parameter
then only the properties specified will be calculated or ignored. You can calculate the default text properties for the TextData object using:

>>> text_data.calculate_builtin_properties()

To learn more about how deepchecks uses properties and how you can calculate or set them yourself, see
the :ref:`Text Properties Guide <nlp__properties_guide>`.

Add Metadata
-------------

You can add metadata to the TextData object:

>>> text_data.set_metadata(metadata_df, categorical_metadata_columns)

To learn more about how deepchecks uses metadata, see the :ref:`Text Metadata Guide <nlp__metadata_guide>`.

Sample
------

You can sample a subset of the TextData object:

>>> text_data.sample(10000)

Working with Class Parameters
---------------------------------

You can work directly with the ``TextData`` object, to inspect its defined raw text, tokenized text, and label:

>>> text_data.raw_text
["This is an example.", "Another example here."]
>>> text_data.tokenized_text
[["This", "is", "an", "example."], ["Another", "example", "here."]]
>>> text_data.label
["positive", "negative"]

Get its internal metadata and properties DataFrames:

>>> text_data.metadata
>>> text_data.properties
