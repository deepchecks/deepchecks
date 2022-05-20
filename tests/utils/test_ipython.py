from unittest.mock import patch

import tqdm
from hamcrest import assert_that, instance_of

from deepchecks.utils import ipython


def test_progress_bar_creation():
    with patch('deepchecks.utils.ipython.is_notebook', return_value=True):
        with patch('deepchecks.utils.ipython.is_widgets_enabled', return_value=True):
            assert_that(
                ipython.create_progress_bar(
                    iterable=list(range(10)),
                    name='Dummy',
                    unit='D'
                ),
                instance_of(tqdm.notebook.tqdm)
            )


def test_progress_bar_creation_with_disabled_widgets():
    with patch('deepchecks.utils.ipython.is_notebook', return_value=True):
        with patch('deepchecks.utils.ipython.is_widgets_enabled', return_value=False):
            assert_that(
                ipython.create_progress_bar(
                    iterable=list(range(10)),
                    name='Dummy',
                    unit='D'
                ),
                instance_of(ipython.PlainNotebookProgressBar)
            )


def test_progress_bar_creation_in_not_notebook_env():
    with patch('deepchecks.utils.ipython.is_notebook', return_value=False):
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
