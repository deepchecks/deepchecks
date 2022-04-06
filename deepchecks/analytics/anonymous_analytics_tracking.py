"""Module for anonymous analytics tracking."""
import http.client
import os
import pathlib
import uuid
import deepchecks

MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
ANALYTICS_DISABLED = os.environ.get("DISABLE_DEEPCHECKS_ANONYMOUS_TRACKING", False)


def send_anonymous_import_event():
    """
    Send an anonymous import event to PostHog.
    """
    if not ANALYTICS_DISABLED:
        try:
            if os.path.exists(os.path.join(MODULE_DIR, '.user_id')):
                with open(os.path.join(MODULE_DIR, '.user_id'), 'r') as f:
                    user_id = f.read()
            else:
                user_id = str(uuid.uuid4())
                with open(os.path.join(MODULE_DIR, '.user_id'), 'w') as f:
                    f.write(user_id)

            conn = http.client.HTTPSConnection('api.deepchecks.com', timeout=3)
            conn.request('GET', f'/metrics?version={deepchecks.__version__}&uuid={user_id}')
            _ = conn.getresponse()
        except Exception:  # pylint: disable=broad-except
            pass
