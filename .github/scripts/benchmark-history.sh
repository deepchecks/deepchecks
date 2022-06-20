#!/bin/bash

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

$ENV_PATH/bin/asv gh-pages