from datetime import datetime
from hamcrest import assert_that, calling, raises, matches_regexp, instance_of
from deepchecks.utils.strings import format_datetime



def test_datetime_instance_format():
    now = datetime.now()
    result = format_datetime(now)
    # %Y/%m/%d %H:%M:%S.%f %Z%z
    pattern = rf"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}"
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
    pattern = rf"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}"
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
    pattern = rf"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}"
    assert_that(
        result,
        instance_of(str),
        matches_regexp(pattern) # type: ignore
    )


def test_format_datetime_with_unsuported_value_type():
    assert_that(
        calling(format_datetime).with_args("hello"),
        raises(ValueError, r"Unsupported value type - str")
    )



