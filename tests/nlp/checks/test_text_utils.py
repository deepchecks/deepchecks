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
"""Test for the text utils module"""
from hamcrest import assert_that, equal_to

from deepchecks.nlp.utils.text import break_to_lines_and_trim


def test_break_to_lines_and_trim():
    text = 'This is a very long text that should be broken into lines. '
    # Text is 59 characters long and has 12 words.

    # Check no change if text is short enough:
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=55, max_line_length=65)
    assert_that(res_text, equal_to(text.strip()))

    # Check that text is broken into lines:
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=15, max_line_length=25)
    assert_that(res_text, equal_to('This is a very long text<br>that should be broken<br>into lines.'))

    # Check that text is trimmed to max_lines:
    res_text = break_to_lines_and_trim(text, max_lines=2, min_line_length=15, max_line_length=25)
    assert_that(res_text, equal_to('This is a very long text<br>that should be broken...'))

    # Check that text with no delimiters is broken in the middle of the line:
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=12, max_line_length=13)
    assert_that(res_text, equal_to('This is a ver-<br>y long text t-<br>hat should be...'))
