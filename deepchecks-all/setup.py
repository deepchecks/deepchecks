#!/usr/bin/env python
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

###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import pathlib
import setuptools
import importlib.util

CURRENT_MODULE = pathlib.Path(__file__).absolute()
SETUP_UTILS_MODULE = CURRENT_MODULE.parent.parent / "setup_utils.py"

spec = importlib.util.spec_from_file_location(
    "deepchecks_setup_utis", 
    SETUP_UTILS_MODULE.absolute()
)

if spec is None:
    raise ImportError("Did not find setup_utils.py module.")

setup_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_utils)
###########################

# TODO: after we move all stuf from '/deepchecks/' to the appropriate 
# submodules then we should move this file to the project root dir

# NOTE: 'deepchecks-all' name has special meaning and is handled by the 
# setup-utils differently then other submodules

setuptools.setup(
    **setup_utils.get_setup_kwargs(submodule="deepchecks-all")
)
