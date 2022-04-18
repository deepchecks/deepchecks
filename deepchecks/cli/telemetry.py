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
"""Telemetry CLI."""
import argparse
import textwrap

from deepchecks.analytics.utils import is_telemetry_enabled, toggle_telemetry


def add_subparser(subparsers, parents) -> None:
    """Add all telemetry tracking parsers.

    Parameters
    ---------
    subparsers: argparse.ArgumentParser
         subparser we are going to attach to
    parents: list
        The list of parent parsers.
    """
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        parents=parents,
        help="Configuration of Deepchecks telemetry reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    telemetry_subparsers = telemetry_parser.add_subparsers()
    telemetry_disable_parser = telemetry_subparsers.add_parser(
        "disable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Disable Deepchecks Telemetry reporting.",
    )
    telemetry_disable_parser.set_defaults(func=disable_telemetry)

    telemetry_enable_parser = telemetry_subparsers.add_parser(
        "enable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable Deepchecks Telemetry reporting.",
    )
    telemetry_enable_parser.set_defaults(func=enable_telemetry)
    telemetry_parser.set_defaults(func=inform_about_telemetry)


def inform_about_telemetry(_: argparse.Namespace) -> None:
    """Inform user about telemetry tracking."""
    is_enabled = is_telemetry_enabled()
    if is_enabled:
        print(
            "Telemetry reporting is currently enabled for this installation."
        )
    else:
        print(
            "Telemetry reporting is currently disabled for this installation."
        )

    print(
        textwrap.dedent(
            """Deepchecks uses telemetry to report anonymous usage information. This information is essential to 
            improve the package for all users."""
        )
    )

    if not is_enabled:
        print("\nYou can enable telemetry reporting using")
        print("\tdeepchecks telemetry enable")
    else:
        print("\nYou can disable telemetry reporting using:")
        print("\tdeepchecks telemetry disable")

    print("\n")


def disable_telemetry(_: argparse.Namespace) -> None:
    """Disable telemetry tracking."""
    toggle_telemetry(False)
    print("Telemetry reporting has been disabled.")


def enable_telemetry(_: argparse.Namespace) -> None:
    """Enable telemetry tracking."""
    toggle_telemetry(True)
    print("Telemetry reporting has been enabled.")
