from deepchecks.analytics.anonymous_telemetry import get_environment_details
from hamcrest import assert_that, equal_to, is_in, has_key


def test_get_environment_details():
    env = get_environment_details()
    assert_that(env, has_key('python_version'))
    assert_that(env, has_key('os'))
    assert_that(env, has_key('deepchecks_version'))
    assert_that(env, has_key('runtime'))

    assert_that(env['runtime'], is_in(['docker', 'colab', 'notebook', 'paperspace', 'native']))
