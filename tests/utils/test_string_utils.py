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
from datetime import datetime

from hamcrest import assert_that, calling, equal_to, instance_of, matches_regexp, raises

from deepchecks.utils.strings import format_datetime, get_ellipsis


def test_get_ellipsis():
    result = get_ellipsis('1234', 3)
    assert_that(result, equal_to('123...'))
    result = get_ellipsis('1234', 4)
    assert_that(result, equal_to('1234'))


def test_datetime_instance_format():
    now = datetime.now()
    result = format_datetime(now)
    # %Y/%m/%d %H:%M:%S.%f %Z%z
    pattern = rf'{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}'
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )


def test_integer_timestamp_format():
    now = datetime.now()
    timestamp = int(now.timestamp())
    result = format_datetime(timestamp)
    # %Y/%m/%d %H:%M:%S.%f %Z%z
    pattern = rf'{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}'
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )


def test_float_timestamp_format():
    now = datetime.now()
    timestamp = now.timestamp()
    result = format_datetime(timestamp)
    # %Y/%m/%d %H:%M:%S.%f %Z%z
    pattern = rf'{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}'
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )


def test_round_timestamp():
    now = datetime.now()
    now = now.replace(second=0, microsecond=0)
    result = format_datetime(now)
    pattern = rf'{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}'
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )

    now = now.replace(hour=0, minute=0)
    pattern = rf'{now.year}/{now.month}/{now.day}'
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )

def test_format_datetime_with_unsupported_value_type():
    assert_that(
        calling(format_datetime).with_args('hello'),
        raises(ValueError, r'Unsupported value type - str')
    )
