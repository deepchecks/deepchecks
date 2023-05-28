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
"""Module of token classification utils for NLP package."""
from collections import Counter
from typing import Dict, List

import numpy as np

__all__ = [
    'clean_iob_prefixes',
    'count_token_classification_labels',
    'annotated_token_classification_text',
]


def clean_iob_prefixes(labels) -> np.array:
    """Remove the initial character of IOB labels (B- and I- and such) if they exist."""
    return np.array([label[2:] if label and label[:2] in ['B-', 'I-', 'O-'] else label for label in labels])


def count_token_classification_labels(labels) -> Dict:
    """Count the number of labels of each kind in a token classification dataset.

    Ignores the initial character of these labels (B- and I- and such) if they exist.
    """
    labels = clean_iob_prefixes(labels)
    return dict(Counter(labels))


def annotated_token_classification_text(token_text, iob_annotations) -> List[str]:
    """Annotate a token classification dataset with IOB tags."""
    annotated_samples = []
    for sample, iob_sample in zip(token_text, iob_annotations):
        annotated_samples.append(' '.join([f'<b>{word}</b>' if iob != 'O' else word for
                                           word, iob in zip(sample, iob_sample)]))
    return annotated_samples
