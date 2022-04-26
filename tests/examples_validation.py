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
import os
import ast
import sys
from urllib.parse import urlparse

import docutils.nodes
from docutils.core import publish_doctree

import deepchecks.tabular.checks as tabular_checks
import deepchecks.vision.checks as vision_checks
from deepchecks import BaseCheck
from deepchecks.utils.strings import generate_check_docs_link


# This script is used to validate the examples in the repo. It will validate that the names of the examples are
# correct and match the name of the check. In addition it will validate the examples include a single H1 tag.

checks_dirs = ["deepchecks/tabular/checks", "deepchecks/vision/checks"]

ignored_files = [
    "deepchecks/vision/checks/distribution/abstract_property_outliers.py",
    ]


def test_read_more_link(check_class, compiled_dir: str):
    link = urlparse(generate_check_docs_link(check_class()))
    # For path "/stable/examples/..." remove the version part
    relevant_path_parts = link.path.split('/')[2:]
    file_path = os.path.join(*compiled_dir.split('/'), *relevant_path_parts)
    if not os.path.exists(file_path):
        print(f'Check {check_class.__name__} does not have a corresponding example file')
        return False
    return True


def get_check_classes_in_module(module):
    all_classes = dir(module)
    for class_name in all_classes:
        class_ = getattr(module, class_name)
        if hasattr(class_, 'mro') and BaseCheck in class_.mro():
            yield class_


def validate_dir(checks_path, examples_path):
    all_valid = True
    for root, _, files in os.walk(checks_path):
        for file_name in files:
            if file_name != "__init__.py" and file_name.endswith(".py"):
                check_path = os.path.join(root, file_name)
                if check_path not in ignored_files:
                    example_file_name = "plot_" + file_name
                    relative_path = "/".join(check_path.split("/")[1:-1])
                    example_path = os.path.join(examples_path, relative_path, "source", example_file_name)
                    if not os.path.exists(example_path):
                        print(f"Check {check_path} does not have a corresponding example file")
                        all_valid = False
                    else:
                        # validate_example(example_path)
                        pass
    return all_valid


# def validate_example(path):
#     with open(path, "r", encoding="utf8") as f:
#         tree = ast.parse(f.read())
#
#     docstring = ast.get_docstring(tree)
#     doctree = publish_doctree(docstring)
#     titles = doctree.traverse(condition=docutils.nodes.title)
#
#     if len(titles) == 0:
#         print(f"Example {path} does not have a single H1 tag")


SOURCE_DIR = "docs/source/examples"
COMPILED_DIR = "docs/build/html"

valid = True
for x in checks_dirs:
    valid = valid and validate_dir(x, SOURCE_DIR)

for check in get_check_classes_in_module(tabular_checks):
    valid = valid and test_read_more_link(check, COMPILED_DIR)

for check in get_check_classes_in_module(vision_checks):
    valid = valid and test_read_more_link(check, COMPILED_DIR)

sys.exit(0 if valid else 1)
