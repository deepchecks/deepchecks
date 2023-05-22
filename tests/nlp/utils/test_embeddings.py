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
"""Test for the embeddings module"""

import numpy as np
from hamcrest import assert_that, equal_to

from deepchecks.nlp.utils.text_embeddings import calculate_builtin_embeddings


def test_simple_embeddings():
    text = ['my name is inigo montoya', 'you killed my father', 'prepare to die']
    embeddings = calculate_builtin_embeddings(np.array(text))
    assert_that(embeddings.shape, equal_to((3, 384)))


def test_edge_cases():
    text = ['!@$', '', None]
    embeddings = calculate_builtin_embeddings(np.array(text))
    assert_that(embeddings.shape, equal_to((3, 384)))
