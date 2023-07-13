.. _nlp__properties_ext:

==========
Properties
==========

Properties are one-dimension values that are extracted from the text. For example, a property could be as simple as
**text length** or **sentiment**, or more complex properties such as the probability that the LLM avoided answering
a question.

Link Validly
------------

The Link Validity property represents the ratio of number links in the text that are valid links, divided by the total
number of links. A valid link is a link that returns a 200 (OK) code when requested. For text without links, the
property will always return 1 (all links valid).

Readability Score
-----------------

A score calculated based on the
`Flesch reading-ease <https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease>`_,
calculated for each text sample. The score typically ranges from 0
(very hard to read, requires intense concentration) to 100 (very easy to read).

Toxicity
--------

The Toxicity property is a measure of how harmful or offensive a text is. The Toxicity property uses a pre-trained model
called `unitary/toxic-bert <https://huggingface.co/unitary/toxic-bert>`_ on HuggingFace, which is based on the BERT
architecture and trained on a large corpus of toxic comments. The model assigns a toxicity score to each text,
ranging from 0 (not toxic) to 1 (very toxic).

Fluency
-------

The Fluency property is a score between 0 and 1 representing how “well” the input text is written, or how close it is
to being a sample of fluent English text. A value of 0 represents very poorly written text, while 1 represents perfectly
written English. The property uses a pre-trained model called
`prithivida/parrot_fluency_model <https://huggingface.co/prithivida/parrot_fluency_model>`_ on HuggingFace.

Formality
---------

The Fluency model returns a measure of how formal the input text is. It uses a pre-traind model called
`s-nlp/roberta-base-formality-ranker <https://huggingface.co/s-nlp/roberta-base-formality-ranker>`_ on HuggingFace,
which was trained to predict for English sentences, whether they are formal or informal, where a score of 0 represents
very informal text, and a score of 1 very formal text.
The model uses the roberta-base architecture, and was trained on
`GYAFC <https://github.com/raosudha89/GYAFC-corpus>`_ from
`Rao and Tetreault, 2018 <https://aclanthology.org/N18-1012>`_ and online formality corpus from
`Pavlick and Tetreault, 2016 <https://aclanthology.org/Q16-1005>`_.

Avoided Answer
--------------

The Avoided Answer property the probability (0 to 1) of how likely it is that the LLM avoided answering a question.
The property uses a pre-trained, bert architecture model that was trained on a dataset of questions and LLM answers
collected from various LLMs, where the model was trained to predict whether the answer is an avoidance or not.

Grounded Score
--------------

The Grounded Score property is a measure of how well the LLM output is grounded in the context of the conversation,
ranging from 0 (not grounded) to 1 (fully grounded).
In the context of this property, grounding means that the LLM output is based on the context given to it as part of the
input, and not on external knowledge, for example knowledge that was present in the LLM training data.

The property is especially useful for evaluating use-cases such as Question Answering, where the LLM is expected to
answer questions based on the context given to it as part of the input, and not based on external knowledge. An example
for such a use-case would be Question Answering based on internal company knowledge, where introduction of external
knowledge (that, for example, may be stale) into the answers is not desired.

The property is calculated by identifying key entities and quantities in the LLM output, such as names, places, dates
and quantities, and then comparing them to the entities and quantities in the input given to the LLM.
The property is calculated as the ratio of the number of entities/quantities in the LLM output that are also in the
input, divided by the total number of entities/quantities in the LLM output.
