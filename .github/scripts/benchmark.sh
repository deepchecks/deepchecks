#!/bin/bash

set -eu

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
    pip install -e .
    git show --no-patch --format="%H (%s)"
    configure_asv
    asv run -E existing --set-commit-hash $(git rev-parse HEAD)
}

pip install asv
asv machine --yes

git update-ref refs/bm/pr HEAD
# We know this is a PR run. The branch is a GitHub refs/pull/*/merge ref, so
# the current target that this PR will be merged into is HEAD^1.
git update-ref refs/bm/merge-target HEAD^1

run_asv

git checkout --force refs/bm/merge-target
run_asv

asv compare refs/bm/merge-target refs/bm/pr