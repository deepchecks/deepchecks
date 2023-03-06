import warnings


class DeprecationHelper(object):
    """
    Wrap a class to warn it is deprecated when called or created, and calls the new class instead.
    This is used to change names to classes and warn users, without breaking backward compatibility.

    Further actions are required to use this class, please see an existing example.

    Parameters
    ----------
    new_target: object,
        the new class to call
    deprecation_message: str
        the message to warn the user with
    """

    def __init__(self, new_target, deprecation_message: str):
        self.new_target = new_target
        self.deprecation_message = deprecation_message

    def _warn(self):
        warnings.warn(self.deprecation_message, DeprecationWarning)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)
