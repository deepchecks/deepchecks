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
"""Module containing junit serializer for the CheckResult type."""
import xml.etree.ElementTree as ET

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import JunitSerializer

__all__ = ['CheckResultSerializer']


class CheckResultSerializer(JunitSerializer['check_types.CheckResult']):
    """Serializes any CheckResult instance into JUnit format.

    Parameters
    ----------
    value : CheckResult
        CheckResult instance that needed to be serialized.
    """

    def __init__(self, value: 'check_types.CheckResult', **kwargs):
        if not isinstance(value, check_types.CheckResult):
            raise TypeError(
                f'Expected "CheckResult" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(self, **kwargs) -> ET.Element:
        """Serialize a CheckResult instance into JUnit format.

        Returns
        -------
        ET.Element
        """
        attributes = {
            'classname': self.value.check.__class__.__module__ + '.' + self.value.check.__class__.__name__
            , 'name': self.value.get_header()
            , 'time': str(self.value.run_time)
        }

        root = ET.Element('testcase', attrib=attributes)
        std_out_element = ET.SubElement(root, 'system-out')
        std_out_element.text = ', '.join([this_result.details for this_result in self.value.conditions_results])

        return root
