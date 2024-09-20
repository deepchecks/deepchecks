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

from deepchecks.utils import ipython
from deepchecks.utils.ipython import HtmlProgressBar

import tqdm
from hamcrest import assert_that, instance_of


def test_progress_bar_creation():
    with patch("deepchecks.utils.ipython.is_zmq_interactive_shell", return_value=True):
        assert_that(
            ipython.create_progress_bar(iterable=list(range(10)), name="Dummy", unit="D"),
            instance_of(ipython.HtmlProgressBar),
        )


def test_progress_bar_creation_in_not_notebook_env():
    with patch("deepchecks.utils.ipython.is_zmq_interactive_shell", return_value=False):
        assert_that(
            ipython.create_progress_bar(iterable=list(range(10)), name="Dummy", unit="D"), instance_of(tqdm.tqdm)
        )


def test_dummy_progress_bar_creation():
    dummy_progress_bar = ipython.DummyProgressBar(name="Dummy")
    assert_that(len(list(dummy_progress_bar.pb)) == 1)


def test_initialization():
    progress_bar = HtmlProgressBar("Test", "unit", range(10), 10)
    assert progress_bar._title == "Test"
    assert progress_bar._unit == "unit"
    assert progress_bar._total == 10
    assert progress_bar._disable is False


def test_iteration_with_disable_true():
    progress_bar = HtmlProgressBar("Test", "unit", range(10), 10, disable=True)
    items = list(progress_bar)
    assert items == list(range(10))
    assert progress_bar._reuse_counter == 1


def test_create_progress_bar_basic():
    result = HtmlProgressBar.create_progress_bar("Test", 5, 10, 2)
    expected = """
            <div>
                <label>
                    Test:<br/>
                    <progress
                        value='5'
                        max='10'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>5/10 [Time: 00:02]</span>
            </div>
        """
    assert result.strip() == expected.strip()


def test_create_progress_bar_with_metadata():
    result = HtmlProgressBar.create_progress_bar("Test", 5, 10, 2, metadata={"key": "value"})
    expected = """
            <div>
                <label>
                    Test:<br/>
                    <progress
                        value='5'
                        max='10'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>5/10 [Time: 00:02, key=value]</span>
            </div>
        """
    assert result.strip() == expected.strip()


def test_create_progress_bar_zero_progress():
    result = HtmlProgressBar.create_progress_bar("Test", 0, 10, 0)
    expected = """
            <div>
                <label>
                    Test:<br/>
                    <progress
                        value='0'
                        max='10'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>0/10 [Time: 00:00]</span>
            </div>
        """
    assert result.strip() == expected.strip()


def test_create_progress_bar_full_progress():
    result = HtmlProgressBar.create_progress_bar("Test", 10, 10, 10)
    expected = """
            <div>
                <label>
                    Test:<br/>
                    <progress
                        value='10'
                        max='10'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>10/10 [Time: 00:10]</span>
            </div>
        """
    assert result.strip() == expected.strip()


def test_create_progress_bar_negative_values():
    result = HtmlProgressBar.create_progress_bar("Test", -1, 10, -1)
    expected = """
            <div>
                <label>
                    Test:<br/>
                    <progress
                        value='-1'
                        max='10'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>-1/10 [Time: 0-1:59]</span>
            </div>
        """
    assert result.strip() == expected.strip()
