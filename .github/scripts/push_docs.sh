#!/bin/bash
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

set -ex


USERNAME="deepchecks-ci"


DOC_REPO="deepchecks.github.io"

cd deepchecks.github.io

GENERATED_DOC_DIR=$GITHUB_WORKSPACE/docs

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f "$GENERATED_DOC_DIR")

if [[ -z "$GITHUB_BASE_REF" ]]
then
	REF="$GITHUB_REF_NAME"
else
	REF="$GITHUB_BASE_REF"
fi

if [[ "$REF" =~ "main" ]]
then
    DIR=dev
else
    # Strip off .X
    DIR="${REF::-2}"
fi

OUTPUT_DIR=testing/$DIR
MSG="Pushing the docs to $OUTPUT_DIR/ for branch: $REF, commit $GITHUB_SHA"

# check if it's a new branch
echo $OUTPUT_DIR > .git/info/sparse-checkout
if ! git show HEAD:$OUTPUT_DIR >/dev/null
then
	# directory does not exist. Need to make it so sparse checkout works
	mkdir -p $OUTPUT_DIR
	touch $OUTPUT_DIR/index.html
	git add $OUTPUT_DIR
fi
git checkout main
git reset --hard origin/main
if [ -d $OUTPUT_DIR ]
then
	git rm -rf $OUTPUT_DIR/ && rm -rf $OUTPUT_DIR/
fi

cp -R $GITHUB_WORKSPACE/docs $OUTPUT_DIR

git config user.email "itay+deepchecks-ci@deepchecks.com"
git config user.name "deepchecks-ci"
git config push.default matching

git add -f $OUTPUT_DIR/
git commit -m "$MSG" $OUTPUT_DIR
git push

echo $MSG