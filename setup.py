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
import pathlib
import re
import typing as t
from functools import lru_cache

import setuptools

DEEPCHECKS = "deepchecks"
SUPPORTED_PYTHON_VERSIONS = '>=3.6, <=3.10'

SETUP_MODULE = pathlib.Path(__file__).absolute()
DEEPCHECKS_DIR = SETUP_MODULE.parent
LICENSE_FILE = DEEPCHECKS_DIR / "LICENSE" 
VERSION_FILE = DEEPCHECKS_DIR / "deepchecks" / "VERSION"
DESCRIPTION_FILE = DEEPCHECKS_DIR / "DESCRIPTION.rst" 


SEMANTIC_VERSIONING_RE = re.compile(
   r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
   r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
   r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
   r"?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


PYTHON_VERSIONING_RE = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?"
    r"(\.dev(0|[1-9][0-9]*))?$"
)


@lru_cache(maxsize=None)
def is_correct_version_string(value: str) -> bool:
    match = PYTHON_VERSIONING_RE.match(value)
    return value == "dev" or match is not None


@lru_cache(maxsize=None)
def get_version_string() -> str:
    if not (VERSION_FILE.exists() and VERSION_FILE.is_file()):
        raise RuntimeError(
            "Version file does not exist! "
            f"(filepath: {str(VERSION_FILE)})")
    else:
        version = VERSION_FILE.open("r").readline()
        if not is_correct_version_string(version):
            raise RuntimeError(
                "Incorrect version string! "
                f"(filepath: {str(VERSION_FILE)})"
            )
        return version


@lru_cache(maxsize=None)
def get_description() -> t.Tuple[str, str]:
    if not (DESCRIPTION_FILE.exists() and DESCRIPTION_FILE.is_file()):
        raise RuntimeError(
            "DESCRIPTION.rst file does not exist! "
            f"(filepath: {str(DESCRIPTION_FILE)})"
        )
    else:
        return (
            "Package for validating your machine learning model and data", 
            DESCRIPTION_FILE.open("r", encoding="utf8").read()
        )


def read_requirements_file(path):
    dependencies = []
    dependencies_links = []
    for line in path.open("r").readlines():
        if "-f" in line or "--find-links" in line:
            dependencies_links.append(
                line
                .replace("-f", "")
                .replace("--find-links", "")
                .strip()
            )
        else:
            dependencies.append(line)
    return dependencies, dependencies_links


@lru_cache(maxsize=None)
def read_requirements() -> t.Dict[str,t.List[str]]:
    requirements_folder = DEEPCHECKS_DIR / "requirements"
    
    if not (requirements_folder.exists() and requirements_folder.is_dir()):
        raise RuntimeError(
            "Cannot find folder with requirements files."
            f"(path: {str(requirements_folder)})"
        )
    else:
        main, main_dep_links = read_requirements_file(requirements_folder / "requirements.txt")
        vision, vision_dep_links = read_requirements_file(requirements_folder / "vision-requirements.txt")
        nlp, nlp_dep_links = read_requirements_file(requirements_folder / "nlp-requirements.txt")

        return {
            "dependency_links": main_dep_links + vision_dep_links,
            "main": main,
            "vision": vision,
            # "nlp": nlp,
        }


# =================================================================================

VERSION = get_version_string()
short_desc, long_desc = get_description()

requirements = read_requirements()
main_requirements = requirements.pop('main')
dependency_links = requirements.pop('dependency_links', [])
extra_requirements = requirements


setuptools.setup(
    # -- description --------------------------------
    name=DEEPCHECKS,
    author='deepchecks',  
    author_email='info@deepchecks.com', 
    version=VERSION,
    description=short_desc,
    long_description=long_desc,
    keywords = ['Software Development', 'Machine Learning'],
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    license_files=('LICENSE', ),
    url = 'https://github.com/deepchecks/deepchecks',
    download_url = "https://github.com/deepchecks/deepchecks/releases/download/{0}/deepchecks-{0}.tar.gz".format(VERSION),
    project_urls={
        'Documentation': 'https://docs.deepchecks.com',
        'Bug Reports': 'https://github.com/deepchecks/deepchecks',
        'Source': 'https://github.com/deepchecks/deepchecks',
        'Contribute!': 'https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md',
    },
    
    # -- dependencies --------------------------------
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    install_requires=main_requirements,
    extras_require=extra_requirements,
    dependency_links=dependency_links,
    include_package_data=True,
)
