.. _nlp__embeddings_guide:

=================
NLP Embeddings
=================

Embeddings are a way to represent text as a vector of numbers. The vector is a representation of the text in the latent
space (or semantic space), in which text with similar meaning is represented by similar vectors.

Embeddings are usually extracted from the one of the final layers of a trained neural network model. This model can either be a
model that was trained on the specific task at hand (e.g. sentiment analysis), or a model that was trained on a
different task, but is known to be good at extracting embeddings (e.g. BERT).


What Are Embeddings Used For?
=============================

Embeddings are used by some of the Deepchecks' checks to produce a meaningful representation of the data, 
insights on the data, since some computations cannot be computed directly on the text (for example, drift).
Inspecting the distribution of the embeddings, or the distance between the embeddings of different texts,
can help uncover potential problems in the way that the datasets were built, or hint about the model's expected
performance on unseen data.

Example for specific scenarios in which using embeddings may come in handy:

#. **Detecting drift in the text** - If the distribution of the embeddings of the training data is different
   from the distribution of the embeddings of the test data, it may indicate that the test data is not
   representative of the training data, and that the model's performance on the test data may be lower than expected.
#. **Investigating low test performance** - By comparing similar texts on which the model doesn't perform well,
    we can try to understand what is the model missing.
    For example, if the model performs well on news articles, but performs poorly on scientific articles,
    it may indicate that the model was trained on a dataset that is biased towards
    the news articles, and that the model is not generalizing well to the scientific articles.
#. **Find conflicting annotations** - Clean data is critical for training a good model. Mistakes in annotations
    (labeling) of the data can lead to a model that is not performing well. By finding similar texts (using embeddings)
    with different annotations, we can find potential annotation mistakes and fix them.


Using Embeddings in Checks
==========================

Whether you are :ref:`Using Deepchecks to Calculate Embeddings` or using your own model's embeddings, the process of
using them in the checks is the same.
In order to use the embeddings of your text in a check, the embeddings should already be part of the ``TextData`` object.


Using Deepchecks to Calculate Embeddings
----------------------------------------

If you don't have model embeddings for you text, you can use deepchecks to calculate the embeddings for you.
deepchecks currently supports using the open-source ``sentence-transformers`` library to calculate the embeddings,
or the paid API of ``open-ai``.

Calculating your embeddings is done by calling the ``calculate_default_embeddings`` method of the ``TextData``
object. This method will calculate the embeddings and add them to the :class:`TextData <deepchecks.nlp.TextData>` object.

Example of calculating the default embeddings in order to use the TextEmbeddingsDrift check:
In the following example, we will calculate the default embeddings in order to use the TextEmbeddingsDrift check:

.. code-block:: python

  from deepchecks.nlp.checks import TextEmbeddingsDrift
  from deepchecks.nlp import TextData

  # Initialize the TextData object
  text_data = TextData(text)

  # Calculate the default embeddings
  text_data.calculate_default_embeddings()

  # Run the check
  TextEmbeddingsDrift().run(text_data)

Note that any use of the ``TextData.calculate_default_embeddings`` method will override the existing embeddings.

Currently, deepchecks supports either using the ``all-MiniLM-L6-v2`` (default) model from the ``sentence-transformers`` library,
or Open AI's ``text-embedding-ada-002`` model. You can choose which model to use by setting the ``model`` parameter
to either ``miniLM`` or ``open_ai``.

The embeddings are automatically saved on a local CSV file so they can be used later. You can change the location and
name of the file by using the ``file_path`` parameter.

.. note:
    If you want to use the Open AI API, you will need to set the ``OPEN_AI_API_KEY`` environment variable to your
    Open AI API key. You can get your API key from the Open AI website.


Using Your Own Embeddings
-------------------------

Whether you saved the deepchecks embeddings for this dataset somewhere to save time, or you used your own model,
you can set the embeddings of the ``TextData`` object to use them by using one of the following methods:

#. When initializing the :class:`TextData <deepchecks.nlp.TextData>` object, pass your pre-calculated
   embeddings to the ``embeddings`` parameter.
#. After the initialization, call the ``set_embeddings`` method of the :class:`TextData <deepchecks.nlp.TextData>`
   object.

In both methods, you can pass the embeddings as a pandas DataFrame, or as a path to a csv file. For the correct format
of the embeddings, see the :ref:`Pre-Calculated Embeddings Format` section.

In the following example, we will pass pre-calculated embeddings to the ``TextData`` object in order to use the
TextPropertyOutliers check:

.. code-block:: python

  from deepchecks.nlp.checks import TextEmbeddingsDrift
  from deepchecks.nlp import TextData

  # Option 1: Initialize the TextData object with the embeddings:
  text_data = TextData(text, embeddings=embeddings)

  # Option 2: Initialize the TextData object and then set the embeddings:
  text_data = TextData(text)
  text_data.set_embeddings(embeddings)

  # Run the check
  TextEmbeddingsDrift().run(text_data)



Pre-Calculated Embeddings Format
################################

The embeddings should be a pandas DataFrame, where each row represents a text sample and each column represents an
embedding dimension. The DataFrame must have the same number of rows as the number of samples in the
:class:`TextData <deepchecks.nlp.TextData>` object, and in the corresponding order.
Note that if you load the embeddings from a csv file, all columns will be loaded and considered as embeddings, so make
sure not to include any other columns in the csv file such as the index column.