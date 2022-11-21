# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing junit serializer for the CheckFailure type."""
import xml.etree.ElementTree as ET

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import JunitSerializer

__all__ = ['CheckFailureSerializer']


FAILURE = 'failure'
SKIPPED = 'skipped'


class CheckFailureSerializer(JunitSerializer['check_types.CheckFailure']):
    """Serializes any CheckFailure instance into JUnit format.

    Parameters
    ----------
    value : CheckFailure
        CheckFailure instance that needed to be serialized.
    """

    def __init__(self, value: 'check_types.CheckFailure', **kwargs):
        if not isinstance(value, check_types.CheckFailure):
            raise TypeError(
                f'Expected "CheckFailure" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(self, failure_tag: str = FAILURE, **kwargs) -> ET.Element:
        """Serialize a CheckFailure instance into JUnit format.

        In general the format is the following. The tag is allowed to be switched out to skipped. Junit output takes the
        following format:

        <failure message="Assertion FAILED: failed assert" type="failure">
            the output of the testcase
        </failure>

        Parameters
        ----------
        failure_tag : str
            Set the tests that fail as either skipped or failure, this allows CI/CD systems to ignore failed tests if
            development does not want them to stop the build.

        Returns
        -------
        ET.Element
        """
        if failure_tag not in [FAILURE, SKIPPED]:
            raise ValueError(f'failure_tag must be one of {FAILURE} or {SKIPPED}')

        attributes = {
            'classname': self.value.check.__class__.__module__ + '.' + self.value.check.__class__.__name__
            , 'name': self.value.get_header()
            , 'time': str(self.value.run_time)
        }

        root = ET.Element('testcase', attrib=attributes)
        attrs = {'type': failure_tag, 'message': str(self.value.exception)}
        failure_element = ET.Element(failure_tag, attrs)
        failure_element.text = self.value.check.metadata().get('summary', '')

        root.append(failure_element)

        return root
