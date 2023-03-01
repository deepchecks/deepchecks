#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#

set -eu
ENV_PATH=$(realpath venv)
PIP_PATH=$ENV_PATH/bin/pip

configure_asv () {
    cat << EOF > asv.conf.json
{
    "version": 1,
    "repo": ".",
    "branches": ["HEAD"],
    "environment_type": "virtualenv",
}
EOF
}

run_asv () {
    $PIP_PATH install -e .
    git show --no-patch --format="%H (%s)"
    configure_asv
    $ENV_PATH/bin/asv run -E existing --set-commit-hash $(git rev-parse HEAD)
}

$PIP_PATH install asv
configure_asv
$ENV_PATH/bin/asv  machine --yes

run_asv

git fetch origin gh-pages:gh-pages
$ENV_PATH/bin/asv gh-pages
