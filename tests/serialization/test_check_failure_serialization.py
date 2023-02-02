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
"""CheckFailure serialization tests."""
import typing as t

import wandb
from hamcrest import (all_of, any_of, assert_that, calling, contains_string, equal_to, has_entry, has_length,
                      has_property, instance_of, only_contains, raises)
from ipywidgets import HTML, VBox

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.serialization.check_failure.html import CheckFailureSerializer as HtmlSerializer
from deepchecks.core.serialization.check_failure.ipython import CheckFailureSerializer as IPythonSerializer
from deepchecks.core.serialization.check_failure.json import CheckFailureSerializer as JsonSerializer
from deepchecks.core.serialization.check_failure.junit import CheckFailureSerializer as JunitSerializer
from deepchecks.core.serialization.check_failure.wandb import CheckFailureSerializer as WandbSerializer
from deepchecks.core.serialization.check_failure.widget import CheckFailureSerializer as WidgetSerializer
from tests.common import DummyCheck, instance_of_ipython_formatter

# =====================================


def test_html_serializer_initialization():
    serializer = HtmlSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_html_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(HtmlSerializer).with_args(5),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "int"')
    )


def test_html_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = HtmlSerializer(failure)
    output = serializer.serialize()

    assert_that(
        output,
        all_of(
            instance_of(str),
            contains_string(str(failure.exception)),
            contains_string(str(failure.header)),
            contains_string(t.cast(str, DummyCheck.__doc__)))
    )


# =====================================


def test_ipython_serializer_initialization():
    serializer = IPythonSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_ipython_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(IPythonSerializer).with_args({}),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "dict"')
    )


def test_ipython_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = IPythonSerializer(failure)
    output = serializer.serialize()

    assert_that(
        output,
        all_of(
            instance_of(list),
            only_contains(instance_of_ipython_formatter()))
    )


# =====================================


def test_json_serializer_initialization():
    serializer = JsonSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_json_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(JsonSerializer).with_args([]),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "list"')
    )


def test_json_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = JsonSerializer(failure)
    output = serializer.serialize()
    assert_json_output(output)


def assert_json_output(output):
    assert_that(
        output,
        all_of(
            instance_of(dict),
            has_entry('header', instance_of(str)),
            has_entry('check', instance_of(dict)))
    )


# =====================================
def test_junit_serializer_initialization():
    serializer = JunitSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_junit_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(JunitSerializer).with_args([]),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "list"')
    )


def test_junit_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = JunitSerializer(failure)
    output = serializer.serialize()

    assert_that(list(output.attrib.keys()), ['classname', 'name', 'time'])
    assert_that(output.tag, 'testcase')

# =====================================

def test_wandb_serializer_initialization():
    serializer = WandbSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_wandb_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WandbSerializer).with_args([]),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "list"')
    )


def test_wandb_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = WandbSerializer(failure)
    output = serializer.serialize()

    assert_that(
        output,
        all_of(
            instance_of(dict),
            has_entry(f'{failure.header}/results', instance_of(wandb.Table)))
    )


# =====================================


def test_widget_serializer_initialization():
    serializer = WidgetSerializer(CheckFailure(
        DummyCheck(),
        Exception("Error"),
        'Failure Header Message'
    ))


def test_widget_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WidgetSerializer).with_args([]),
        raises(
            TypeError,
            'Expected "CheckFailure" but got "list"')
    )


def test_widget_serialization():
    failure = CheckFailure(DummyCheck(), ValueError("Check Failed"), 'Failure Header Message')
    serializer = WidgetSerializer(failure)
    output = serializer.serialize()

    assert_that(
        output,
        all_of(
            instance_of(VBox),
            has_property(
                'children',
                all_of(
                    instance_of(tuple),
                    has_length(3),
                    only_contains(instance_of(HTML)))))
    )
    assert_that(
        output.children[0],
        has_property(
            'value',
            all_of(
                instance_of(str),
                contains_string(failure.header)))
    )
    assert_that(
        output.children[1],
        has_property(
            'value',
            all_of(
                instance_of(str),
                contains_string(t.cast(str,DummyCheck.__doc__))))
    )
    assert_that(
        output.children[2],
        has_property(
            'value',
            all_of(
                instance_of(str),
                contains_string(str(failure.exception))))
    )
