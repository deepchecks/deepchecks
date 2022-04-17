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
"""CLI for Deepchecks."""
import argparse
import logging
import platform
import sys

from deepchecks.cli import telemetry

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments."""
    parser = argparse.ArgumentParser(
        prog="deepchecks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deepchecks command line interface. Deepchecks is a Python package for comprehensively validating "
                    "your machine learning models and data with minimal effort.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print installed Deepchecks version",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Deepchecks commands")

    telemetry.add_subparser(subparsers, parents=parent_parsers)

    return parser


def print_version() -> None:
    """Print version information of deepchecks and python."""
    try:
        from deepchecks import __version__  # pylint: disable=import-outside-toplevel

        dc_version = __version__
    except ModuleNotFoundError:
        dc_version = None

    print(f"Deepchecks Version :         {dc_version}")
    print(f"Python Version     :         {platform.python_version()}")
    print(f"Operating System   :         {platform.platform()}")
    print(f"Python Path        :         {sys.executable}")


def main():
    """Represent the main entry point of the cli."""
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    if hasattr(cmdline_arguments, "func"):
        cmdline_arguments.func(cmdline_arguments)
    elif hasattr(cmdline_arguments, "version"):
        print_version()
    else:
        # user has not provided a subcommand, let's print the help
        logger.error("No command specified.")
        arg_parser.print_help()
        sys.exit(1)
