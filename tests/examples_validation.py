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
import docutils.nodes
from docutils.core import publish_doctree

# This script is used to validate the examples in the repo. It will validate that the names of the examples are
# correct and match the name of the check. In addition it will validate the examples include a single H1 tag.

checks_dirs = ["deepchecks/tabular/checks", "deepchecks/vision/checks"]

ignored_files = [
    "deepchecks/vision/checks/distribution/abstract_property_outliers.py",
    ]


def validate_dir(checks_path, examples_path):
    for root, subdirs, files in os.walk(checks_path):
        for file_name in files:
            if file_name != "__init__.py" and file_name.endswith(".py"):
                check_path = os.path.join(root, file_name)
                if check_path not in ignored_files:
                    example_file_name = "plot_" + file_name
                    relative_path = "/".join(check_path.split("/")[1:-1])
                    example_path = os.path.join(examples_path, relative_path, "source", example_file_name)
                    if not os.path.exists(example_path):
                        print("Check {} does not have a corresponding example file".format(check_path))
                    else:
                        validate_example(example_path)


def validate_example(path):
    with open(path, "r", encoding="utf8") as f:
        tree = ast.parse(f.read())

    docstring = ast.get_docstring(tree)
    doctree = publish_doctree(docstring)
    titles = doctree.traverse(condition=docutils.nodes.title)

    if len(titles) != 1:
        print("Example {} does not have a single H1 tag".format(path))


for x in checks_dirs:
    validate_dir(x, "docs/source/examples")
