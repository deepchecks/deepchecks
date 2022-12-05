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
"""Module containing JUnit serializer for the SuiteResult type."""
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Union

from six import u

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import JunitSerializer
from deepchecks.core.serialization.check_failure.junit import FAILURE, SKIPPED, CheckFailureSerializer
from deepchecks.core.serialization.check_result.junit import CheckResultSerializer

__all__ = ['SuiteResultSerializer']


def _clean_illegal_xml_chars(string_to_clean):
    """Remove any illegal unicode characters from the given XML string.

    @see: http://stackoverflow.com/questions/1707890/fast-way-to-filter-illegal-xml-unicode-chars-in-python
    """
    illegal_unichrs = [
        (0x00, 0x08),
        (0x0B, 0x1F),
        (0x7F, 0x84),
        (0x86, 0x9F),
        (0xD800, 0xDFFF),
        (0xFDD0, 0xFDDF),
        (0xFFFE, 0xFFFF),
        (0x1FFFE, 0x1FFFF),
        (0x2FFFE, 0x2FFFF),
        (0x3FFFE, 0x3FFFF),
        (0x4FFFE, 0x4FFFF),
        (0x5FFFE, 0x5FFFF),
        (0x6FFFE, 0x6FFFF),
        (0x7FFFE, 0x7FFFF),
        (0x8FFFE, 0x8FFFF),
        (0x9FFFE, 0x9FFFF),
        (0xAFFFE, 0xAFFFF),
        (0xBFFFE, 0xBFFFF),
        (0xCFFFE, 0xCFFFF),
        (0xDFFFE, 0xDFFFF),
        (0xEFFFE, 0xEFFFF),
        (0xFFFFE, 0xFFFFF),
        (0x10FFFE, 0x10FFFF),
    ]

    illegal_ranges = [f'{chr(low)}-{chr(high)}' for (low, high) in illegal_unichrs if low < sys.maxunicode]

    illegal_xml_re = re.compile(u('[%s]') % u('').join(illegal_ranges))
    return illegal_xml_re.sub('', string_to_clean)


class SuiteResultSerializer(JunitSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into Junit format.

    Parameters
    ----------
    value : SuiteResult
        SuiteResult instance that needed to be serialized.
    """

    def __init__(self, value: 'suite.SuiteResult', **kwargs):
        if not isinstance(value, suite.SuiteResult):
            raise TypeError(
                f'Expected "SuiteResult" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(
        self,
        failure_tag: str = 'failure',
        encoding: str = 'utf-8',
        return_xml: bool = False,
        **kwargs
    ) -> Union[str, ET.Element]:
        """Serialize a SuiteResult instance into Junit format. This can then be output as a str or a XML object.

        Parameters
        ----------
        failure_tag : str
            Set the tests that fail as either skipped or failure, this allows CI/CD systems to ignore failed tests if
            development does not want them to stop the build.
        encoding : str
            What format to encode the output str into.
        return_xml : bool
            Whether to return a str or a XML object
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        Union[str, ET.Element]
        """
        if failure_tag not in [FAILURE, SKIPPED]:
            raise ValueError(f'failure_tag must be one of {FAILURE} or {SKIPPED}')

        results = self._serialize_test_cases(encoding, failure_tag)

        root = self._create_junit_test_suites_wrapper(failure_tag, results)

        root = self._process_test_suites(failure_tag, results, root)

        if return_xml:
            return root

        else:
            xml_str = _clean_illegal_xml_chars(ET.tostring(root, encoding=encoding).decode(encoding))
            return xml_str

    @staticmethod
    def _process_test_suites(failure_tag: str, results: Dict, root: ET.Element) -> ET.Element:
        """Organize the results into test suites and gather metadata about each one of the tests.

        Parameters
        ----------
        failure_tag : str
            Set the tests that fail as either skipped or failure, this allows CI/CD systems to ignore failed tests if
            development does not want them to stop the build.
        results : dict
            Organized and serialized test suites to be added to the root of the xml tree.
        root : ET.Element
            The root of the xml tree to append test suites and test cases too.

        Returns
        -------
        ET.Element
        """
        run_time = 0

        for this_suite in results.keys():
            attributes = {
                'name': this_suite
                , 'errors': '0'
                , 'tests': str(len(results[this_suite]))
                , 'timestamp': datetime.now().replace(microsecond=0).isoformat()
            }

            if failure_tag == f'{FAILURE}':
                attributes.update({'failures': str(
                    sum(list(results[this_suite][i])[0].tag == FAILURE for i in range(len(results[this_suite]))))})
            elif failure_tag == f'{SKIPPED}':
                attributes.update({'skipped': str(
                    sum(list(results[this_suite][i])[0].tag == SKIPPED for i in range(len(results[this_suite]))))})
                attributes.update({'failures': '0'})

            suite_time = sum([int(this_result.attrib['time']) for this_result in results[this_suite]])
            run_time += suite_time

            attributes.update({'time': str(suite_time)})

            test_suite = ET.SubElement(root, 'testsuite', attrib=attributes)
            test_suite.extend(results[this_suite])

        root.attrib['time'] = str(run_time)

        return root

    def _create_junit_test_suites_wrapper(self, failure_tag, results) -> ET.Element:
        """Create the root node of the XML output.

        Parameters
        ----------
        failure_tag : str
            Set the tests that fail as either skipped or failure, this allows CI/CD systems to ignore failed tests if
            development does not want them to stop the build.
        results : dict
            Organized and serialized test suites to be added to the root of the xml tree.

        Returns
        -------
        ET.Element
        """
        count = sum([len(value) for key, value in results.items()])

        attributes = {
            'name': self.value.name
            , 'errors': '0'
            , 'tests': str(count)
        }

        if failure_tag == f'{FAILURE}':
            attributes.update({'failures': str(len(self.value.failures))})
        elif failure_tag == f'{SKIPPED}':
            attributes.update({'skipped': str(len(self.value.failures))})
            attributes.update({'failures': '0'})

        root = ET.Element('testsuites', attrib=attributes)

        return root

    def _serialize_test_cases(self, encoding, failure_tag) -> Dict[str, List[ET.Element]]:
        """Iterate over the test cases and serialize them into test suites.

        Parameters
        ----------
        encoding : str
            What format to encode the output str into.
        failure_tag : str
            Set the tests that fail as either skipped or failure, this allows CI/CD systems to ignore failed tests if
            development does not want them to stop the build.

        Returns
        -------
        Dict[str, List[ET.Element]]
        """
        results = {}

        for it in self.value.results:
            if isinstance(it, check_types.CheckResult):
                result = CheckResultSerializer(it).serialize(encoding=encoding)
            elif isinstance(it, check_types.CheckFailure):
                result = CheckFailureSerializer(it).serialize(encoding=encoding, failure_tag=failure_tag)
            else:
                raise TypeError(f'Unknown result type - {type(it)}')

            if 'checks' in result.attrib['classname'].split('.'):
                test_suite_name = result.attrib['classname'].split('.')[
                    result.attrib['classname'].split('.').index('checks') + 1]
            else:
                test_suite_name = 'checks'

            if test_suite_name not in results:
                results[test_suite_name] = [result]
            else:
                results[test_suite_name].append(result)

        return results
