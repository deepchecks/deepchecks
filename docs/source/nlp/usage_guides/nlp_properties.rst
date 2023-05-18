.. _nlp__properties_guide:

=================
NLP Properties
=================

Properties are one-dimension values that are extracted from the text. For example, a property could be **text length**
or **sentiment**.
Deepchecks includes :ref:`built-in properties <Deepchecks' Built-in Properties>` and supports :ref:`using your own
properties <Using Your Own Properties>`.

Not to be confused with :ref:`metadata <nlp__metadata_guide>`, which is additional data that comes with it organically,
such as the text's author or date of creation.


What Are Properties Used For?
=============================

Properties are used by some of the Deepchecks' checks, in order to extract meaningful
features from the data, since some computations cannot be computed directly on the text (for example, drift).
Inspecting the distribution of the property's values (e.g. to notice some texts are extremely long,
or that the most common language in the corpus is different between the train and test sets) can help uncover potential
problems in the way that the datasets were built, or hint about the model's expected performance on unseen data.

Example for specific scenarios in which measuring properties may come in handy:

#. **Investigating low test performance** - detecting high drift in certain properties may help you pinpoint the causes
   of the model's lower performance on the test data.
#. **Generalizability on new data** - a drift in significant data properties,
   may indicate lower ability of the model to accurately predict on the new (different) unlabeled data.
#. **Find weak segments** - The properties can be used to segment the data and test for low performing segments.
   If found, the weak segment may indicate an underrepresented segment or an area where the data quality is worse.
#. **Find obscure relations between the data and the targets** - the model training might be affected
   by properties we are not aware of, and that aren't the core attributes of what we are aiming for it to learn.
   For example, in a classification dataset of true and false statements, if only true facts are written in detail,
   and false facts are written in a short and vague manner, the model might learn to predict the label by the length
   of the statement, and not by the actual content. In this case, the model will perform well on the training data,
   and may even perform well on the test data, but will fail to generalize to new data.


Deepchecks' Built-in Properties
===============================

You can either use the built-in properties or implement your own ones and pass them to the relevant checks.
The built-in image properties are:

==============================  ==========
Property name                   What is it
==============================  ==========
Text Length                     Number of characters in the text
Average Word Length             Average number of characters in a word
Max Word Length                 Maximum number of characters in a word
% Special Characters            Percentage of special characters in the text
Language                        Language of the text. Uses the langdetect library
Sentiment                       Sentiment of the text. Uses the textblob library
Subjectivity                    Subjectivity of the text. Uses the textblob library
Toxicity*                       Toxicity of the text. Uses the unitary/toxic-bert model
Fluency*                        Fluency of the text. Uses the prithivida/parrot_fluency_model model
Formality*                      Formality of the text. Uses the s-nlp/roberta-base-formality-ranker model
Lexical Density                 Percentage of unique words in the text, rounded up to 2 decimal digits
Unique Noun Count*              Number of unique noun words in the text
Readability Score               A score calculated based on Flesch reading-ease per text sample. For more information: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
Average Sentence Length         Average number of words per sentence in the text
Count Unique URLs               Number of unique URLS per text sample.
Count Unique Email Address      Number of unique email addresses per text sample.
Count Unique Syllables          Number of unique syllables per text sample.
Reading Time                    Time taken in seconds to read a text sample.
Sentence Length                 Number of sentences per text sample.
Average Syllable Length         Average number of syllables per sentence per text sample.
==============================  ==========

*These properties are not calculated by default, as they may take a long time to calculate. To use them, pass
``include_long_calculation_properties=True`` to the :class:`TextData.calculate_properties <deepchecks.nlp.TextData>` method.


Using Properties in Checks
==========================

Whether you are using the built-in properties or your own, the process of using them in the checks is the same.
In order to use the properties of your text in a check, the properties should already be part of the ``TextData`` object.


Calculating The Built-in Properties
-----------------------------------

In order to use the built-in properties, you must call the ``calculate_builtin_properties`` method of the ``TextData``
object. This method will calculate the properties and add them to the :class:`TextData <deepchecks.nlp.TextData>` object.

Example of calculating the built-in properties in order to use the TextPropertyOutliers check:
In the following example, we will calculate the default properties in order to use the TextPropertyOutliers check:

.. code-block:: python

  from deepchecks.nlp.checks import TextPropertyOutliers
  from deepchecks.nlp import TextData

  # Initialize the TextData object
  text_data = TextData(text)

  # Calculate the default properties
  text_data.calculate_builtin_properties()

  # Run the check
  TextPropertyOutliers().run(text_data)

Note that any use of the ``TextData.calculate_builtin_properties`` method will override the existing properties.

Including or Ignoring Properties
#################################

When calculating the properties, you can choose to include or exclude specific properties, by passing the
``include_properties`` or ``ignore_properties`` parameters to the ``calculate_builtin_properties`` method.
The parameters should be a list of the names of the properties to include or ignore. Note that only one of the
parameters can be passed to the method.

In the following example, we will calculate the built-in properties and ignore the ``Text Length`` property:

.. code-block:: python

  text_data.calculate_builtin_properties(ignore_properties=['Text Length'])


Moreover, some properties are not calculated by default, as they may take a long time to calculate. In order to
use them, pass ``include_long_calculation_properties`` to the ``calculate_builtin_properties`` method.

In the following example, we will calculate the properties and include only the long calculation property "Toxicity":

.. code-block:: python

  text_data.calculate_builtin_properties(include_long_calculation_properties=True, include_properties=['Toxicity'])

Saving The Calculated Properties
################################

If you want to save the calculated properties, you can use the ``save_properties`` method of the ``TextData`` object:

.. code-block:: python

  text_data.save_properties('path/to/file.csv')

See how to reload the properties in the :ref:`Using Your Own Properties` section.


Using Your Own Properties
-------------------------

Whether you saved the deepchecks properties for this dataset somewhere to save time, or you calculated something smart
of your own, you can set the properties of the ``TextData`` object to be your own, by using one of the following methods:

#. When initializing the :class:`TextData <deepchecks.nlp.TextData>` object, pass your pre-calculated
   properties to the ``properties`` parameter.
#. After the initialization, call the ``set_properties`` method of the :class:`TextData <deepchecks.nlp.TextData>`
   object.

In both methods, you can pass the properties as a pandas DataFrame, or as a path to a csv file. For the correct format
of the properties, see the :ref:`Pre-Calculated Properties Format` section.

Additionally, it's advised to also use the ``categorical_properties`` parameter to specify which properties are
categorical. The parameter should be a list of the names of the categorical properties (columns).

In the following example, we will pass pre-calculated properties to the ``TextData`` object in order to use the
TextPropertyOutliers check:

.. code-block:: python

  from deepchecks.nlp.checks import TextPropertyOutliers
  from deepchecks.nlp import TextData

  # Option 1: Initialize the TextData object with the properties:
  text_data = TextData(text, properties=properties, categorical_properties=categorical_properties)

  # Option 2: Initialize the TextData object and then set the properties:
  text_data = TextData(text)
  text_data.set_properties(properties, categorical_properties)

  # Run the check
  TextPropertyOutliers().run(text_data)



Pre-Calculated Properties Format
################################

The properties should be a pandas DataFrame, where each row represents a text sample and each column represents a
property. The DataFrame must have the same number of rows as the number of samples in the
:class:`TextData <deepchecks.nlp.TextData>` object, and in the corresponding order.
Note that if you load the properties from a csv file, all columns will be loaded and considered as properties, so make
sure not to include any other columns in the csv file such as the index column.