.. _nlp_supported_tasks:

===================================
Working with Labels and Predictions
===================================

Some checks, mainly the ones related to model evaluation, require labels and model predictions in order to run.
In the deepchecks nlp package, predictions are passed into the suite / check ``run`` method as pre-computed
predictions only (passing a fitted model is currently not supported).


.. _nlp_supported_tasks__types:

Supported Task Types
====================

Deepchecks currently supports two NLP task types:

* **Text Classification**: Text classification is any NLP task in which a whole body of text (ranging from a sentence
  to a document) is assigned a class (in the binary/multiclass case) or a certain set of classes (in the multilabel
  case). In both the binary, the multiclass and the multilabel case the class "belongs" / "classifies" the whole text
  sample. Examples for such tasks are: Sentiment Analysis, Topic Extraction, harmful content detection and more.
* **Token Classification**: Token Classification is any NLP task in which each word (or to be more accurate, token) in
  the text sample is assigned a class of its own. In many cases most tokens will belong to a "background" class,
  allowing the model to focus on the interesting tokens. Examples for such tasks are: Named Entity Recognition, Part-of-speech annotation (in which all tokens have a non-background class).

.. _nlp_supported_labels__predictions_format:

Supported Labels and Predictions Format
=======================================

While labels are passed when constructing the :class:`TextData <text_data.TextData>` object, predictions are passed
separately to the ``run`` method of the check / suite. Labels and predictions must be in the format detailed in this
section, according to the task type.

Text Classification
-------------------

Label Format
~~~~~~~~~~~~

* For text classification the accepted label format differs between multilabel and
  single label cases. For single label data, the label should be passed as a sequence of labels, with one entry
  per sample that can be either a string or an integer. For multilabel data, the label should be passed as a
  sequence of sequences, with the sequence for each sample being a binary vector, representing the presence of
  the i-th label in that sample.

>>> text_classification_label_multiclass = ['class_0', 'class_0', 'class_1', 'class_2']
>>> text_classification_label_multilabel = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0]]

.. note::

    For multilabel tasks, in order for deepchecks to use string names for the different classes (rather than just noting
    the classes id in the label matrix) you may pass a list of the class names to the ``classes`` argument
    of the :class:`TextData <text_data.TextData>` constructor method. This list of names, having the same length as the
    number of rows in the label matrix, will be used to name the multilabel classes throughout deepchecks.

Prediction Format
~~~~~~~~~~~~~~~~~

.. note::

    Class probabilities (and for multilabel tasks, also predictions) are always provided as a matrix of
    (n_samples, n_classes). In order to understand which column corresponds to each of the class names present in the
    labels and the predictions, this matrix must follow the convention that the i-th element represents the class
    probabilities for the class in the i-th position in the sorted array of class names. The sorted array of class names
    is the result of sorting the set of all class names present in the label and prediction, namely
    ``sorted(list(set(y_true).union(set(y_pred))))``.

Single Class Predictions
""""""""""""""""""""""""

* **predictions** - A sequence of class names or indices with one entry per sample, matching the set of classes
  present in the labels.
* **probabilities** - A sequence of sequences with each element containing the vector of class probabilities for
  each sample. Each such vector should have one probability per class according to the class (sorted) order, and
  the probabilities should sum to 1 for each sample.

>>> predictions = ['class_1', 'class_1', 'class_2']
>>> # Note that even in the binary case the probability must be specified for each class, as is the case in this example
>>> probabilities = [[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]

Multilabel Predictions
""""""""""""""""""""""

* **predictions** - A sequence of sequences with each element containing a binary vector denoting the presence of
  the i-th class for the given sample. Each such vector should have one binary indicator per class according to
  the class (sorted) order. More than one class can be present for each sample.
* **probabilities** - A sequence of sequences with each element containing the vector of class probabilities for
  each sample. Each such vector should have one probability per class according to the class (sorted) order, and
  the probabilities should range from 0 to 1 for each sample, but are not required to sum to 1.

>>> predictions = [[0, 0, 1], [0, 1, 1]]
>>> probabilities = [[0.2, 0.3, 0.8], [0.4, 0.9, 0.6]]

Token Classification
--------------------

Label Format
~~~~~~~~~~~~

* For token classification the accepted label format is a sequence of sequences,
  with the inner sequence containing tuples in the following format: (class_name, span_start, span_end).
  span_start and span_end are the start and end character indices of the token within the text, as it was
  passed to the raw_text argument. Each outer sequence contains the sequence of tokens for each sample.

>>> token_classification_label = [[('class_1', 0, 2), ('class_2', 7, 10)], [('class_2', 42, 54)], []]

Prediction Format
~~~~~~~~~~~~~~~~~

* **predictions** - A sequence of sequences, with the inner sequence containing tuples in the following
  format: (class_name, span_start, span_end, class_probability). span_start and span_end are the start and end
  character indices  of the token within the text, as it was passed to the raw_text argument. Each upper level
  sequence contains a sequence of tokens for each sample.
* **probabilities** - No probabilities should be passed for Token Classification tasks. Passing probabilities will
  result in an error.

>>> predictions = [[('class_1', 0, 2, 0.8), ('class_2', 7, 10, 0.9)], [('class_2', 42, 54, 0.4)], []]

..
    external links to open in new window

.. |sequence| raw:: html

    <a href="https://www.pythontutorial.net/advanced-python/python-sequences/#:~:text=A%20sequence%20is%20a%20positionally,s%5Bn%2D1%5D%20." target="_blank">sequence</a>
