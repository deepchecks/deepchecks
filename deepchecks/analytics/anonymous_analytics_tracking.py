"""Module for anonymous analytics tracking."""
import os
import pathlib
import uuid
import posthog
import deepchecks

posthog.project_api_key = 'phc_r4IgrazCbgeYovXw1iwxl3elpHzoyl2B4mGIMt20ntb'
posthog.host = 'https://app.posthog.com'
posthog.debug = True

MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
ANALYTICS_DISABLED = not os.environ.get("DEEPCHECKS_ANONYMOUS_TRACKING", True)


def send_anonymous_import_event():
    """
    Send an anonymous import event to PostHog.
    """
    if ANALYTICS_DISABLED:
        try:
            if os.path.exists(os.path.join(MODULE_DIR, '.user_id')):
                with open(os.path.join(MODULE_DIR, '.user_id'), 'r') as f:
                    user_id = f.read()
            else:
                user_id = str(uuid.uuid4())
                with open(os.path.join(MODULE_DIR, '.user_id'), 'w') as f:
                    f.write(user_id)

            posthog.capture(distinct_id=user_id,
                            event='package-import',
                            properties={'version': deepchecks.__version__})
            posthog.flush()

        except Exception:  # pylint: disable=broad-except
            pass
