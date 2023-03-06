import warnings


class DeprecationHelper(object):
    def __init__(self, new_target, deprecation_message):
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
