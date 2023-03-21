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
"""Module of text utils for NLP package."""

__all__ = ['break_to_lines_and_trim']


def break_to_lines_and_trim(s, max_lines: int = 10, min_line_length: int = 50, max_line_length: int = 60):
    """Break a string to lines and trim it to a maximum number of lines.

    Parameters
    ----------
    s : str
        The string to break.
    max_lines : int, default 10
        The maximum number of lines to return.
    min_line_length : int, default 50
        The minimum length of a line.
    max_line_length : int, default 60
        The maximum length of a line.
    """
    separating_delimiters = [' ', '\t', '\n', '\r']
    lines = []
    for i in range(max_lines):  # pylint: disable=unused-variable
        if len(s) < max_line_length:  # if remaining string is short enough, add it and break
            lines.append(s.strip())
            break
        else:  # find the first delimiter from the end of the line
            max_line_length = min(max_line_length, len(s)-1)
            for j in range(max_line_length, min_line_length-1, -1):
                if s[j] in separating_delimiters:
                    lines.append(s[:j])
                    s = s[j:].strip()
                    break
            else:  # if no delimiter was found, break in the middle of the line
                lines.append(s[:max_line_length].strip() + '-')
                s = s[max_line_length:].strip()
    else:  # if the loop ended without breaking, and there is still text left, add an ellipsis
        if len(s) > 0:
            lines[-1] = lines[-1] + '...'
    return '<br>'.join(lines)
