.. _nlp_textdata_object:

==================
The TextData Object
==================

The ``TextData`` is a container for the your text data, labels, and relevant metadata for NLP tasks and is a basic
building block in the deepchecks.nlp subpackage.
In order to use any of the deepchecks.nlp subpackage functionality, you need to create a ``TextData`` object first.
The TextData object enables easy access to metadata, embeddings and properties relevant for training and validating ML
models.

Class Properties
==================

The main properties are:

- **raw_text** - The raw text data, a list of strings representing the raw text of each sample. Each sample can be a
  sentence, paragraph, or a document, depending on the task.
- **tokenized_text** - The tokenized text data, a sequence of sequences of strings representing the tokenized text
  of each sample. Serves as an alternative for the raw text data, and is the recommended way to input your text
  for token classification tasks.
- **label** - The labels for the text data.
- **task_type** - The task type, see the :ref:`Supported Tasks Guide <nlp_supported_tasks>` for more information.

TextData API Reference
.. currentmodule:: deepchecks.nlp.text_data

.. autosummary::

    TextData

Creating a TextData
=======================

The default ``TextData`` constructor expects to get a sequence of raw text strings or tokenized text.
The rest of the arguments are optional, but if your data has labels you would want to define them,
as many checks require the dataset labels in order to run.

.. admonition:: Defining task_type
   :class: attention

   If you define labels, you must also define the ``task_type`` so deepchecks will know how to parse the labels.


>>> raw_text = ["This is an example.", "Another example here."]
>>> tokenized_text = [["This", "is", "an", "example."], ["Another", "example", "here."]]
>>> labels = ["positive", "negative"]
>>> task_type = "text_classification"
>>> text_data = TextData(raw_text=raw_text, label=labels, task_type=task_type)
>>> # Alternatively using tokenized text:
>>> text_data = TextData(tokenized_text=tokenized_text, label=labels, task_type=task_type)

If you have tokenized text, you can also create a TextData object from it:

>>> text_data = TextData(tokenized_text=tokenized_text, label=labels, task_type=task_type)

If you're running deepchecks on a token classification task and you have passed ``raw_text`` to the constructor,
deepchecks will break the text into tokens for you, using ``.split()`` to split the text into tokens.

Useful Functions
===================

Calculate Default Properties
-----------------------------

You can calculate the default text properties for the TextData object:

>>> text_data.calculate_default_properties()

To learn more about how deepchecks uses properties and how you can calculate or set them yourself, see
the :ref:`Text Properties Guide <nlp_properties_guide>`.

Add Metadata
-------------

You can add metadata to the TextData object:

>>> text_data.set_metadata(metadata_df, categorical_metadata_columns)

To learn more about how deepchecks uses metadata, see the :ref:`Text Metadata Guide <nlp_metadata_guide>`.

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
