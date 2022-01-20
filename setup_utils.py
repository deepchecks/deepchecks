# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import typing as t
import re
import shutil
import pathlib
import setuptools
from functools import lru_cache

DEEPCHECKS = "deepchecks"
SUPPORTED_PYTHON_VERSIONS = '>=3.6, <=3.10'

SETUP_UTILS_MODULE = pathlib.Path(__file__)
DEEPCHECKS_DIR = SETUP_UTILS_MODULE.parent
LICENSE_FILE = DEEPCHECKS_DIR / "LICENSE" 
VERSION_FILE = DEEPCHECKS_DIR / "VERSION" 
DESCRIPTION_FILE = DEEPCHECKS_DIR / "DESCRIPTION.rst" 


SEMANTIC_VERSIONING_RE = re.compile(
   r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
   r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
   r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
   r"?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@lru_cache(maxsize=None)
def is_correct_version_string(value: str) -> bool:
   match = SEMANTIC_VERSIONING_RE.match(value)
   return match is not None


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


def create_version_file(destination: pathlib.Path):
   if not (destination.exists() and destination.is_dir()):
      raise RuntimeError(
         "Do not know where to put the __version__.py file,"
         f"'distination'(path: {str(destination)}) does not exist or is not a dir!"
      )
   else:
      version = get_version_string().replace("\n", "").strip()
      with (destination / "__version__.py").open("w") as f:
         f.write(f"version = '{version}'")


def copy_license_file(destination: pathlib.Path):
   if not (LICENSE_FILE.exists() and LICENSE_FILE.is_file()):
      raise RuntimeError(f"LICENSE file does not exist! (filepath: {str(LICENSE_FILE)})")
   elif not (destination.exists() and destination.is_dir()):
      raise RuntimeError(
         "Do not know where to copy the LICENSE file,"
         f"'distination'(path: {str(destination)}) does not exist or is not a dir!"
      )
   else:
      shutil.copy2(LICENSE_FILE.absolute(), (destination / "LICENSE").absolute())
      

@lru_cache(maxsize=None)
def get_description() -> t.Tuple[str, str]:
   if not (DESCRIPTION_FILE.exists() and DESCRIPTION_FILE.is_file()):
      raise RuntimeError(
         "DESCRIPTION.rst file does not exist! "
         f"(filepath: {str(DESCRIPTION_FILE)})"
      )
   else:
      return "Deepchecks package", DESCRIPTION_FILE.open("r").read()


@lru_cache(maxsize=None)
def read_requirements_file(path: pathlib.Path) -> t.List[str]:
   if not (path.exists() and path.is_file()):
      raise RuntimeError(
         "provided requirements file path does not exist "
         f"or is not a file! (filepath: {str(path)})"
      )
   else:
      return path.open("r").read().splitlines()


def verify_submodule_existence(submodule: str) -> pathlib.Path:
   submodule_path = DEEPCHECKS_DIR / submodule

   if not (
      (submodule_path / "src" / DEEPCHECKS / submodule).exists()
      and (submodule_path / "src" / DEEPCHECKS / submodule).is_dir()
      and (submodule_path / "tests").exists()
      and (submodule_path / "tests").is_dir()
      and (submodule_path / "setup.py").exists()
      and (submodule_path / "setup.py").is_file()
   ):
      raise RuntimeError(
         "Are you sure that you provided correct submodule name? "
         f"Path {submodule_path.absolute()} does not exist or it`s directory "
         "structure is incorrect! Directory structure of a submodule must "
         "look like this:\n\n"
         "  + submodule-name\n"
         "     + src\n"
         "        + deepchecks\n"
         "           + submodule-name\n"
         "              + <src files>:\n"
         "     + tests/\n"
         "     + setup.py\n"
         "     + requirements.txt\n"
      )
   
   return submodule_path

t.NewType
def get_setup_kwargs(submodule: str, **kwargs) -> t.Dict[str, t.Any]:
   version = get_version_string()
   description, long_description = get_description()
   
   if submodule == "deepchecks-all":
      sumbodule_path = DEEPCHECKS_DIR / "deepchecks-all"
      requirements = []
      extras_requirements = {
         "core": "deepchecks.core=={version}",
         "tabular": "deepchecks.tabular=={version}",
         "vision": "deepchecks.vision=={version}",
      }
      copy_license_file(sumbodule_path)
      name = DEEPCHECKS
   else:
      sumbodule_path = verify_submodule_existence(submodule)
      requirements = read_requirements_file(sumbodule_path / "requirements.txt")
      extras_requirements = {}
      create_version_file(sumbodule_path / "src" / DEEPCHECKS / submodule)
      copy_license_file(sumbodule_path)
      name = f"{DEEPCHECKS}.{submodule}"

   download_url = (
      "https://github.com/deepchecks/deepchecks/"
      "releases/download/{0}/deepchecks-{0}.tar.gz"
   ).format(version)

   setup_kwargs = dict(
      # general info
      name = name,
      version = version,
      author = 'deepchecks',
      author_email = 'info@deepchecks.com',
      project_urls={
         'Documentation': 'https://docs.deepchecks.com',
         'Bug Reports': 'https://github.com/deepchecks/deepchecks',
         'Source': 'https://github.com/deepchecks/deepchecks',
         'Contribute!': 'https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md',
      },
      download_url = download_url,
      description = description,
      long_description = long_description,
      license_files = ('LICENSE',),
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

      # Package config
      # packages = setuptools.find_packages('src'),
      packages = setuptools.find_namespace_packages(where="src", include=[DEEPCHECKS]),
      package_dir = {'': 'src'},
      package_data = {DEEPCHECKS: ['LICENSE']},
      # namespace_packages = [DEEPCHECKS],
      include_package_data = True,
      install_requires = requirements,
      extras_require = extras_requirements,
      python_requires = SUPPORTED_PYTHON_VERSIONS,
   )

   setup_kwargs.update(kwargs)
   return setup_kwargs