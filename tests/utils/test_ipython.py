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
"""IPython utilities tests."""
from unittest.mock import patch

import tqdm
from hamcrest import assert_that, instance_of

from deepchecks.utils import ipython


def test_progress_bar_creation():
    with patch('deepchecks.utils.ipython.is_zmq_interactive_shell', return_value=True):
        assert_that(
            ipython.create_progress_bar(
                iterable=list(range(10)),
                name='Dummy',
                unit='D'
            ),
            instance_of(ipython.HtmlProgressBar)
        )


def test_progress_bar_creation_in_not_notebook_env():
    with patch('deepchecks.utils.ipython.is_zmq_interactive_shell', return_value=False):
        assert_that(
            ipython.create_progress_bar(
                iterable=list(range(10)),
                name='Dummy',
                unit='D'
            ),
            instance_of(tqdm.tqdm)
        )


def test_dummy_progress_bar_creation():
    dummy_progress_bar = ipython.DummyProgressBar(name='Dummy')
    assert_that(len(list(dummy_progress_bar.pb)) == 1)
