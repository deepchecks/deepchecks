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
"""Package for common static resources."""

__all__ = ['DEEPCHECKS_STYLE', 'DEEPCHECKS_HTML_PAGE_STYLE']


DEEPCHECKS_STYLE = """
table.deepchecks {
    border: none;
    border-collapse: collapse;
    border-spacing: 0;
    color: black;
    font-size: 12px;
    table-layout: fixed;
    width: max-content;
}
table.deepchecks thead {
    border-bottom: 1px solid black;
    vertical-align: bottom;
}
table.deepchecks tr,
table.deepchecks th, 
table.deepchecks td {
    text-align: right;
    vertical-align: middle;
    padding: 0.5em 0.5em;
    line-height: normal;
    white-space: normal;
    max-width: none;
    border: none;
}
table.deepchecks th {
    font-weight: bold;
}
table.deepchecks tbody tr:nth-child(odd) {
    background: white;
}
table.deepchecks tbody tr:nth-child(even) {
    background: #f5f5f5;
}
table.deepchecks tbody tbody tr:hover {
    background: rgba(66, 165, 245, 0.2);
}
details.deepchecks {
    border: 1px solid #d6d6d6;
    margin-bottom: 0.25rem;
}
details.deepchecks > div {
    display: flex;
    flex-direction: column;
    padding: 1rem 1.5rem 1rem 1.5rem;
}
details.deepchecks > summary {
    display: list-item;
    background-color: #f9f9f9;
    font-weight: bold;
    padding: 0.5rem;
}
details[open].deepchecks > summary {
    border-bottom: 1px solid #d6d6d6;
}
div.deepchecks-tabs {
    width: 100%;
    display: flex;
    flex-direction: column;
}
div.deepchecks-tabs-btns {
    width: 100%;
    height: fit-content;
    display: flex;
    flex-direction: row;
}
div.deepchecks-tabs-btns > button {
    margin: 0;
    background-color: #f9f9f9;
    border: 1px solid #d6d6d6;
    padding: .5rem 1rem .5rem 1rem;
    cursor: pointer;
}
div.deepchecks-tabs-btns > button:focus {
    box-shadow: none;
    outline: none;
}
div.deepchecks-tabs-btns > button[open] {
    background-color: white;
    border-bottom: none;
    border-top: 2px solid #1975FA;
}
div.deepchecks-tabs > div.deepchecks-tab {
    display: None;
}
div.deepchecks-tabs > div.deepchecks-tab[open] {
    display: flex;
    flex-direction: column;
    border: 1px solid #d6d6d6;
    padding: 1rem;
}
"""


DEEPCHECKS_HTML_PAGE_STYLE = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 16px;
    line-height: 1.5;
    color: #212529;
    text-align: left;
    margin: auto;
    background-color: white; 
    padding: 1rem 1rem 0 1rem;
}
%deepchecks-style
""".replace('%deepchecks-style', DEEPCHECKS_STYLE)


