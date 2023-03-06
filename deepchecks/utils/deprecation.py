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
"""Module helping with renaming classes."""
import warnings


class DeprecationHelper(object):
    """
    Wrap a class to warn it is deprecated when called or created, and calls the new class instead.

    This is used to change names to classes and warn users, without breaking backward compatibility.

    Further actions are required to use this class, please see an existing example.

    """

    def __init__(self, new_target, deprecation_message: str):
        """
        Init class.

        Parameters
        ----------
        new_target: object,
            the new class to call
        deprecation_message: str
            the message to warn the user with
        """
        self.new_target = new_target
        self.deprecation_message = deprecation_message

    def _warn(self):
        warnings.warn(self.deprecation_message, DeprecationWarning)

    def __call__(self, *args, **kwargs):
        """
        Handle calls of the object.

        Parameters
        ----------
        args args
        kwargs kwargs

        Returns instance of the new class
        -------

        """
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        """
        Handle usage of attributes of the object.

        Parameters
        ----------
        attr The attribute

        Returns The attribute
        -------

        """
        self._warn()
        return getattr(self.new_target, attr)
