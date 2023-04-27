# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module with common docstrings for the nlp package."""
from deepchecks.utils.decorators import Substitution

_shared_docstrings = {}

_shared_docstrings['prediction_formats'] = """
Notes
-----
The accepted formats for providing model predictions and probabilities are detailed below

**Text Classification**

*Single Class Predictions*

- predictions - A sequence of class names or indices with one entry per sample, matching the set of classes
  present in the labels.
- probabilities - A sequence of sequences with each element containing the vector of class probabilities for
  each sample. Each such vector should have one probability per class according to the class (sorted) order, and
  the probabilities should sum to 1 for each sample.

*Multilabel Predictions*

- predictions - A sequence of sequences with each element containing a binary vector denoting the presence of
  the i-th class for the given sample. Each such vector should have one binary indicator per class according to
  the class (sorted) order. More than one class can be present for each sample.
- probabilities - A sequence of sequences with each element containing the vector of class probabilities for
  each sample. Each such vector should have one probability per class according to the class (sorted) order, and
  the probabilities should range from 0 to 1 for each sample, but are not required to sum to 1.

**Token Classification**

- predictions - A sequence of sequences, with the inner sequence containing tuples in the following
  format: (class_name, span_start, span_end, class_probability). span_start and span_end are the start and end
  character indices  of the token within the text, as it was passed to the raw_text argument. Each upper level
  sequence contains a sequence of tokens for each sample.
- probabilities - No probabilities should be passed for Token Classification tasks. Passing probabilities will
  result in an error.

Examples
--------

**Text Classification**

*Single Class Predictions*

>>> predictions = ['class_1', 'class_1', 'class_2']
>>> probabilities = [[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]

*Multilabel Predictions*

>>> predictions = [[0, 0, 1], [0, 1, 1]]
>>> probabilities = [[0.2, 0.3, 0.8], [0.4, 0.9, 0.6]]

**Token Classification**

>>> predictions = [[('class_1', 0, 2, 0.8), ('class_2', 7, 10, 0.9)], [('class_2', 42, 54, 0.4)], []]
""".strip('\n')


_shared_docstrings['text_normalization_params'] = """
ignore_case: bool, default True
    ignore text case during samples comparison.
remove_punctuation: bool, default True
    ignore punctuation characters during samples comparison.
normalize_unicode: bool, default True
    normilize unicode characters before samples comparison.
remove_stopwords: bool, default True
    remove stopwrods before samples comparison.
ignore_whitespace: bool, default False
    ignore whitespace characters during samples comparison.
""".strip('\n')


_shared_docstrings['max_text_length_for_display_param'] = """
max_text_length_for_display : int, default 30
    truncate text samples to given length before display.
""".strip('\n')


docstrings = Substitution(**_shared_docstrings)
