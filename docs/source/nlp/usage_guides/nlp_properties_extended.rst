.. _nlp__properties_ext:

==========
Properties
==========

Properties are one-dimensional values that are calculated on each text sample. For example, a property could be simple
text characteristics such as the number of words in the text, or more complex properties such identifying if the
text contains toxic language.

* `Link Validity <#link-validity>`__
* `Readability Score <#readability-score>`__
* `Toxicity <#toxicity>`__
* `Fluency <#fluency>`__
* `Formality <#formality>`__
* `Avoided Answer <#avoided-answer>`__
* `Grounded in Context <#grounded-in-context>`__
* `Relevance <#relevance>`__

Link Validity
-------------

The Link Validity property represents the ratio of number links in the text that are valid links, divided by the total
number of links. A valid link is a link that returns a **200 OK** HTML status when sent a HTTP HEAD request. For text
without links, the property will always return 1 (all links valid).

`Back to Property List <#properties>`__

Readability Score
-----------------

A score calculated based on the
`Flesch reading-ease <https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease>`__,
calculated for each text sample. The score typically ranges from 0
(very hard to read, requires intense concentration) to 100 (very easy to read) for english text, though in theory the
score can range from -inf to 206.835 for arbitrary strings.

`Back to Property List <#properties>`__

Toxicity
--------

The Toxicity property is a measure of how harmful or offensive a text is. The Toxicity property uses a pre-trained model
called `unitary/toxic-bert <https://huggingface.co/unitary/toxic-bert>`__ on HuggingFace, which is created and
maintained by the `Detoxify <https://github.com/unitaryai/detoxify>`__ team, based on the BERT
architecture and trained on a large corpus of toxic comments. The model assigns a toxicity score to each text,
ranging from 0 (not toxic) to 1 (very toxic).

`Back to Property List <#properties>`__

Examples
~~~~~~~~

================================  ========
Text                              Toxicity
================================  ========
Hello! How can I help you today?  0.0007
You have been a bad user!         0.301
I hate you!                       0.951
================================  ========

Fluency
-------

The Fluency property is a score between 0 and 1 representing how “well” the input text is written, or how close it is
to being a sample of fluent English text. A value of 0 represents very poorly written text, while 1 represents perfectly
written English. The property uses a pre-trained model called
`prithivida/parrot_fluency_model <https://huggingface.co/prithivida/parrot_fluency_model>`__ on HuggingFace, which
was created by the authors of the `Parrot Paraphraser <https://github.com/PrithivirajDamodaran/Parrot_Paraphraser>`__
library.

`Back to Property List <#properties>`__

Examples
~~~~~~~~

===============================================================================================================================================================  ========
Text                                                                                                                                                             Fluency
===============================================================================================================================================================  ========
Natural language processing is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence.                                      0.97
Pass on what you have learned. Strength, mastery, hmm… but weakness, folly, failure, also. Yes, failure, most of all. The greatest teacher, failure is.          0.75
Whispering dreams, forgotten desires, chaotic thoughts, dance with words, meaning elusive, swirling amidst.                                                      0.2
===============================================================================================================================================================  ========

Formality
---------

The Formality model returns a measure of how formal the input text is. It uses a pre-traind model called
`s-nlp/roberta-base-formality-ranker <https://huggingface.co/s-nlp/roberta-base-formality-ranker>`__ on HuggingFace,
trained by the Skolkovo Institute of Science and Technology (s-nlp).
The model was trained to predict for English sentences, whether they are formal or informal, where a score of 0
represents very informal text, and a score of 1 very formal text.
The model uses the roberta-base architecture, and was trained on
`GYAFC <https://github.com/raosudha89/GYAFC-corpus>`__ from
`Rao and Tetreault, 2018 <https://aclanthology.org/N18-1012>`__ and online formality corpus from
`Pavlick and Tetreault, 2016 <https://aclanthology.org/Q16-1005>`__.

`Back to Property List <#properties>`__

Examples
~~~~~~~~

================================================================  ========
Text                                                              Formality
================================================================  ========
I hope this email finds you well                                  0.79
I hope this email find you swell                                  0.28
What's up doc?                                                    0.14
================================================================  ========

Avoided Answer
--------------

The Avoided Answer property calculates the probability (0 to 1) of how likely it is that the LLM avoided answering a
question.
The property uses a pre-trained bert architecture model that was trained on a dataset of questions and LLM answers
collected from various LLMs, where the model was trained to predict whether the answer is an avoidance or not.

`Back to Property List <#properties>`__

Examples
~~~~~~~~

============================================================================  ========
Question                                                                      Avoided Answer
============================================================================  ========
The answer is 42                                                              0.001
You should ask the appropriate authorities                                    0.681
As a Large Language Model trained by Open AI, I can not answer this question  0.994
============================================================================  ========

Grounded in Context
-------------------

The Grounded in Context Score is a measure of how well the LLM output is grounded in the context of the conversation,
ranging from 0 (not grounded) to 1 (fully grounded).
In the definition of this property, grounding means that the LLM output is based on the context given to it as part of
the input, and not on external knowledge, for example knowledge that was present in the LLM training data. Grounding
can be thought of as kind of hallucination, as hallucination are outputs that are not grounded in the context, nor
are true to the real world.

The property is especially useful for evaluating use-cases such as Question Answering, where the LLM is expected to
answer questions based on the context given to it as part of the input, and not based on external knowledge. An example
for such a use-case would be Question Answering based on internal company knowledge, where introduction of external
knowledge (that, for example, may be stale) into the answers is not desired - we can imagine a case in which an LLM is
asked a question about the company's revenue, and the answer is based on the company's internal financial reports, and
not on external knowledge such as the company's Wikipedia page. In the context of Question Answering, any answer that
is not grounded in the context can be considered a hallucination.

The property is calculated by identifying key entities and quantities in the LLM output, such as names, places, dates
and prices, and then identifying the same entities and quantities in the input given to the LLM.
The property is calculated as the ratio of the number of entities/quantities in the LLM output that are also in the
input, divided by the total number of entities/quantities in the LLM output.

`Back to Property List <#properties>`__

Examples
~~~~~~~~

======================================================================================================================  =====================================================  ===================
LLM Input                                                                                                               LLM Output                                             Grounded in Context
======================================================================================================================  =====================================================  ===================
Michael Jordan (1963) is an American former professional basketball player and businessman. In what year was he born?   He was born in 1963.                                   1.0
Michael Jordan (1963) is an American former professional basketball player and businessman. When was Michael born?      Michael Jeffrey Jordan was born in 1963                0.5
Michael Jordan (1963) is an American former professional basketball player and businessman. What did he achieve?        He won many NBA championships with the Chicago Bulls   0.0
======================================================================================================================  =====================================================  ===================


Relevance
---------

The Relevance score is a measure of how relevant the LLM output is to the input, ranging from 0 (not relevant) to 1
(fully relevant).
In a Question Answering use-case, a relevant output is an output that addresses the question that was asked. It may not
be a full answer - the model may reply that it cannot answer that, doesn't know, or may even be wrong; But an answer
is considered relevant if the model understood the question and gave an answer that is based on that understanding,
regardless of the correctness or the completeness of the answer.

The Relevance Score is calculated by taking the maximum of 2 similar "tests", that try to measure how relevant the output is.
The reason the maximum of these 2 tests is taken is that we aim to find at least 1 positive proof that the output is relevant,
as these two tests are more prone to false negatives than false positives.

The first test is the Mean Max Word Similarity. The test takes each meaningful word in the user's input (excluding "stop words",
which are words that are not meaningful, such as "the", "a", "and", etc.), and looks for the closest word in the LLM output.
Then, it takes the similarity score between the input word and the output word, and averages the scores for all the words
in the input. The similarity score is the cosine similarity between the word embeddings of the input word and the output word.
For instance, if the user's input is "What is the color of the sky?", and the LLM replies "The sky is blue", then the
meaningful words in the input are "color" and "sky". The word "sky" in the input will be matched to the word "sky" in
the output (with a similarity score of 1, as it is the same word), and the word "color" in the input will be matched to
the word "blue" in the output (with a high similarity score of e.g. 0.9, as the words are semantically similar).
The average of the similarity scores will be the Mean Max Word Similarity score.

The second test is the Lowest Max Entity Similarity. The test takes each entity in the user's input, and looks for the
closest entity in the LLM output. Then, it takes the similarity score between the input entity and the output entity,
and returns the minimum the scores for all the entities in the input. The similarity score is the cosine similarity between the
entity embeddings of the input entity and the output entity.
For instance, if the user's input is "Where was the first World War II attack in the United States?", and the LLM replies "Pearl Harbour",
then the entities in the input are "World War II" and "United States", and the only entity in the output is "Pearl Harbour".
Both "World War II" and "United States" in the input will be matched to "Pearl Harbour" in the output with high similarity scores (e.g. 0.8 and 0.9),
and the Lowest Max Entity Similarity score will be the minimum of the two scores (0.8).

The reason for taking the mean of the Max Word Similarity and the minimum of the Max Entity Similarity is that for
entities we expect to find a high match for each one, or else the output is not relevant enough. However, for regular
words, we expect to find a match for only some of them, and the rest can be ignored.

`Back to Property List <#properties>`__

Examples
~~~~~~~~
======================================================================================================================  =====================================================  ===================
User Input                                                                                                               LLM Output                                             Grounded in Context
======================================================================================================================  =====================================================  ===================
What is the color of the sky?                                                                                           The sky is blue                                        0.81
Where was the first World War II attack in the United States?                                                           Pearl Harbor                                           0.84
======================================================================================================================  =====================================================  ===================
