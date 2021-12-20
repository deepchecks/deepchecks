#!/bin/sh
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

ipynb_files=$(find docs/source/examples/howto-guides -type f -name "*.ipynb")

for path in $ipynb_files
do
  jupyter nbconvert --to notebook --execute "$path" --stdout > "$path".new && mv "$path".new "$path"
done