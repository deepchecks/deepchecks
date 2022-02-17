# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import typing as t
import inspect
import os
import sys
import pathlib
import functools
from subprocess import check_output


# -- Path setup --------------------------------------------------------------

CURRENT_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR.absolute()))

from deepchecks.utils.strings import to_snake_case

with open(os.path.join(PROJECT_DIR, 'VERSION')) as version_file:
    VERSION = version_file.read().strip()

# -- Project information -----------------------------------------------------


project = 'Deepchecks'
copyright = '2021, Deepchecks'
author = 'Deepchecks'
is_readthedocs = os.environ.get("READTHEDOCS")
version = os.environ.get("READTHEDOCS_VERSION") or VERSION
language = os.environ.get("READTHEDOCS_LANGUAGE")

GIT = {
    "user": "deepchecks",
    "repo": "deepchecks"
}

try:
    GIT["branch"] = tag = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    GIT["release"] = release = check_output(['git', 'describe', '--tags', '--always']).decode().strip()
except Exception as error:
    raise RuntimeError("Failed to extract commit hash!") from error

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#
extensions = [
    'nbsphinx',

    # by itself sphinx_gallery is not able to work with jupyter notebooks
    # but nbsphinx extension is able to use some of it functionality to create
    # thumbnail galleries. Link to the doc - https://nbsphinx.readthedocs.io/en/0.8.7/subdir/gallery.html
    #
    'sphinx_gallery.load_style',
    #

    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'sphinx_search.extension',
    'sphinx.ext.autosectionlabel',
]

# Add any paths that contain templates here, relative to this directory.
#
templates_path = ['_templates']

# A list of warning types to suppress arbitrary warning messages.
# List of all possible types: https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=suppress_warnings#confval-suppress_warnings
#
suppress_warnings = [
    # to ignore messages like:
    # <module-file-path>.py:docstring of <routine-name>:: WARNING: py:class reference target not found: package.foo.Bar
    #
    "ref.class",
    "ref.exc",
    # to ignore messages like:
    # WARNING: duplicate label <string>, other instance in <doc-path>.rst
    #
    "autosectionlabel.*"
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, Sphinx will warn about all references where the target cannot be found.
# Default is False. You can activate this mode temporarily using the -n command-line switch.
#
nitpicky = True

# A boolean that decides whether module names are prepended to all object names.
# Default is True.
#
add_module_names = False

# If true, suppress the module name of the python reference if it can be resolved.
# The default is False.
#
python_use_unqualified_type_names = True


# -- autosummary settings --------------------------------------------------

# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each. It is enabled by default.
#
# autosummary_generate = False

# If true, autosummary overwrites existing files by generated stub pages.
# Defaults to true (enabled).
#
autosummary_generate_overwrite = False

# A boolean flag indicating whether to document classes and
# functions imported in modules. Default is False
#
autosummary_imported_members = False

# If False and a module has the __all__ attribute set, autosummary
# documents every member listed in __all__ and no others. Default is True
#
autosummary_ignore_module_all = False

# A dictionary of values to pass into the template engine’s context
# for autosummary stubs files.
#
autosummary_context = {'to_snake_case': to_snake_case}

# TODO: explaine
autosummary_filename_map = {
    "deepchecks.tabular.checks": "../deepchecks.tabular.checks",
    "deepchecks.vision.checks": "../deepchecks.vision.checks",
}

# -- autodoc settings --------------------------------------------------

# Autodoc settings.
# This value selects how the signature will be displayed for the class defined by autoclass directive.
# The possible values are:
# + "mixed"     - Display the signature with the class name (default).
# + "separated" - Display the signature as a method.
#
autodoc_class_signature = 'separated'

# This value controls how to represent typehints. The setting takes the following values:
#    'signature' – Show typehints in the signature (default)
#    'description' – Show typehints as content of the function or method The typehints of overloaded functions or methods will still be represented in the signature.
#    'none' – Do not show typehints
#    'both' – Show typehints in the signature and as content of the function or method
#
autodoc_typehints = 'signature'

# This value controls the format of typehints.
# The setting takes the following values:
#   + 'fully-qualified' – Show the module name and its name of typehints
#   + 'short' – Suppress the leading module names of the typehints (default in version 5.0)
#
autodoc_typehints_format = 'short'

# True to convert the type definitions in the docstrings as references. Defaults to False.
#
napoleon_preprocess_types = False

# Report warnings for all validation checks
numpydoc_validation_checks = {"all"}

# -- nbsphinx extension settings --------------------------------------------------

nbsphinx_prolog = r"""

.. raw:: html

    <div style="
        width: 100%;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-end;
    ">
        <span style="white-space: nowrap; margin-right: 10px;">
            <a
                href="{{  env.config.generate_binder_url(env.docname) }}"
                style="vertical-align:text-bottom"
                target="_blank" rel="noopener noreferrer">
                <img
                    alt="Binder badge"
                    {% if env.config.html_context["is_readthedocs"] %}
                    src="/{{ env.config.html_context["language"] }}/{{ env.config.html_context["version"] }}/_static/binder-badge.svg"
                    {% else %}
                    src="/_static/binder-badge.svg"
                    {% endif %}
                    style="vertical-align:text-bottom">
            </a>
        </span>
        <span style="white-space: nowrap; margin-right: 10px;">
            <a
                href="{{  env.config.generate_colab_url(env.docname) }}"
                style="vertical-align:text-bottom"
                target="_blank" rel="noopener noreferrer">
                <img
                    alt="Colab badge"
                    {% if env.config.html_context["is_readthedocs"] %}
                    src="/{{ env.config.html_context["language"] }}/{{ env.config.html_context["version"] }}/_static/colab-badge.svg"
                    {% else %}
                    src="/_static/colab-badge.svg"
                    {% endif %}
                    style="vertical-align:text-bottom">
            </a>
        </span>
    </div>

{% set apipath = env.config.get_check_example_api_reference(env.docname) %}

{% if apipath %}
{{ apipath }}
{% endif %}

"""

nbsphinx_requirejs_options = {
    "src": "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
    "integrity": "sha512-VCK7oF67GXNc+J7zsu5o57jtxhLA75nSMHGaq8Q8TCOxDj4nMDw5dhQZvm9Cd9RN+3zgcodqbKcRc9gyPP8a2w==",
    "crossorigin": "anonymous"
}
nbsphinx_assume_equations = False


# -- Copybutton settings --------------------------------------------------

# Only copy lines starting with the input prompts,
# valid prompt styles: [
#     Python Repl + continuation (e.g., '>>> ', '... '),
#     Bash (e.g., '$ '),
#     ipython and qtconsole + continuation (e.g., 'In [29]: ', '  ...: '),
#     jupyter-console + continuation (e.g., 'In [29]: ', '     ...: ')
# ]
# regex taken from https://sphinx-copybutton.readthedocs.io/en/latest/#using-regexp-prompt-identifiers
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Continue copying lines as long as they end with this character
copybutton_line_continuation_character = "\\"


# -- Linkcode ----------------------------------------------------------------

def _import_object_from_name(module_name, fullname):
    obj = sys.modules.get(module_name)
    if obj is None:
        return None
    for comp in fullname.split('.'):
        try:
            obj = getattr(obj, comp)
        except AttributeError:
            pass
    return obj


_top_modules = ['deepchecks']
_source_root = None


def _find_source_root(source_abs_path):
    # Note that READTHEDOCS* environment variable cannot be used, because they
    # are not set under the CI environment.
    global _source_root
    if _source_root is not None:
        return _source_root

    assert os.path.isfile(source_abs_path)
    dirname = os.path.dirname(source_abs_path)
    while True:
        parent = os.path.dirname(dirname)
        if os.path.basename(dirname) in _top_modules:
            _source_root = parent
            return _source_root
        if len(parent) == len(dirname):
            raise RuntimeError(
                'Couldn\'t parse root directory from '
                'source file: {}'.format(source_abs_path))
        dirname = parent


def _get_source_relative_path(source_abs_path):
    return os.path.relpath(source_abs_path, _find_source_root(source_abs_path))


def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    # Import the object from module path
    obj = _import_object_from_name(info['module'], info['fullname'])

    # If it's not defined in the internal module, return None.
    mod = inspect.getmodule(obj)

    if mod is None:
        return None
    if not mod.__name__.split('.')[0] in _top_modules:
        return None

    # Get the source file name and line number at which obj is defined.
    try:
        filename = inspect.getsourcefile(obj)
    except TypeError:
        # obj is not a module, class, function, ..etc.
        return None
    except AttributeError:
        return None
    # inspect can return None for cython objects
    if filename is None:
        return None

    # Get the source line number
    _, linenum = inspect.getsourcelines(obj)
    assert isinstance(linenum, int)

    filename = os.path.realpath(filename)
    relpath = _get_source_relative_path(filename)
    return f'https://github.com/deepchecks/deepchecks/blob/{tag}/{relpath}#L{linenum}'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#
html_static_path = ['_static']

html_css_files = ['css/custom.css',]

#
html_sidebars = {
    "**": ["sidebar-nav-bs"]
}

# Path to logo and favicon
#
html_logo = "./_static/deepchecks_logo.svg"
html_favicon = "./_static/favicons/favicon.ico"

# If true, the reST sources are included in the HTML build as _sources/name. The default is True.
#
html_copy_source = True

# Theme specific options.
# See documentation of the 'pydata_sphinx_theme' theme
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html#adding-external-links-to-your-nav-bar
#
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 6,
    "navbar_end": ["search-field", "navbar-icon-links", "menu-dropdown", ],
    # "page_sidebar_items": ["page-toc", "create-issue", "show-page-source"],
    "page_sidebar_items": ["page-toc", ],
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": f"https://github.com/{GIT['user']}/{GIT['repo']}",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://deepcheckscommunity.slack.com/join/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg#/shared-invite/email",
            "icon": "fab fa-slack",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/deepchecks/",
            "icon": "fab fa-python",
        }
    ],
}

# vars that will be passed to the jinja templates during build
#
html_context = {
    "version": version,
    "is_readthedocs": is_readthedocs,
    "language": language,
}


# -- Other -------------------------------------------------


def get_report_issue_url(pagename: str) -> str:
    template = (
        "https://github.com/{user}/{repo}/issues/new?title={title}&body={body}&labels={labels}"
    )
    return template.format(
        user=GIT["user"],
        repo=GIT["repo"],
        title="[Docs] Documentation contains a mistake.",
        body=f"Package Version: {version};\nPage: {pagename}",
        labels="labels=chore/documentation",
    )


def generate_colab_url(notebook_path: str) -> t.Optional[str]:
    notebook_path = notebook_path.replace(".txt", "")
    notebook_path = notebook_path if notebook_path.endswith(".ipynb") else notebook_path + ".ipynb"
    notebook_name = notebook_path.split("/")[-1]

    if not is_example_notebook(notebook_name):
        return

    template = (
        "https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{notebook_path}"
    )

    return template.format(
        user=GIT['user'],
        repo=GIT['repo'],
        branch=GIT['release'],
        notebook_path=f"docs/source/{notebook_path}"
    )


def generate_binder_url(notebook_path: str) -> t.Optional[str]:
    notebook_path = notebook_path.replace(".txt", "")
    notebook_path = notebook_path if notebook_path.endswith(".ipynb") else notebook_path + ".ipynb"
    notebook_name = notebook_path.split("/")[-1]

    if not is_example_notebook(notebook_name):
        return

    template = (
        "https://mybinder.org/v2/gh/{user}/{repo}/{branch}?labpath={filepath}"
    )

    return template.format(
        user=GIT['user'],
        repo=GIT['repo'],
        branch=GIT['release'],
        filepath=f"docs/source/{notebook_path}"
    )


@functools.lru_cache(maxsize=None)
def get_example_notebooks() -> t.Tuple[pathlib.Path, ...]:
    examples_folder = PROJECT_DIR / "docs/source/examples"

    if not examples_folder.exists() or not examples_folder.is_dir():
        raise RuntimeError("Did not find the folder with the example notebooks.")

    return tuple(examples_folder.glob("**/*.ipynb"))


@functools.lru_cache(maxsize=None)
def snake_case_to_camel_case(val: str) -> str:
    return "".join(it.capitalize() for it in val.split("_") if it)


def is_example_notebook(filepath: str) -> bool:
    notebook_name = filepath.split("/")[-1].replace(".txt", "")
    notebooks = {it.name for it in get_example_notebooks()}
    return notebook_name in notebooks


def get_check_example_api_reference(filepath: str) -> t.Optional[str]:
    if not (
        filepath.startswith("docs/source/examples/checks/")
        or filepath.startswith("examples/checks/")
    ):
        return

    notebook_name = snake_case_to_camel_case(
        filepath.split("/")[-1]
            .replace(".txt", "")
            .replace(".ipynb", "")
            .replace(".py", "")
    )
    type = filepath.split("/")[2]

    import deepchecks.tabular.checks
    import deepchecks.vision.checks
    tabular_check_clazz = getattr(deepchecks.tabular.checks, notebook_name, None)
    vision_check_clazz = getattr(deepchecks.vision.checks, notebook_name, None)

    mapping = {
        'tabular': tabular_check_clazz,
        'vision': vision_check_clazz
    }
    if type not in mapping or mapping[type] is None or not hasattr(mapping[type], "__module__"):
        return

    clazz_module = ".".join(mapping[type].__module__.split("."))

    apipath = f"/api/generated/{clazz_module}.{notebook_name}"
    result = f"* :doc:`API Reference - {notebook_name} <{apipath}>`"
    return result


# -- Registration of hooks ---------


def setup(app):

    def add_custom_routines(app, pagename, templatename, context, doctree):
        context["get_report_issue_url"] = get_report_issue_url
        context["generate_colab_url"] = generate_colab_url
        context["generate_binder_url"] = generate_binder_url
        context["is_example_notebook"] = is_example_notebook
        context["get_example_notebooks"] = get_example_notebooks

    # make custom routines available within html templates
    app.connect("html-page-context", add_custom_routines)

    # make next functions available within nbsphinx prolog/epilog templates
    app.config.generate_binder_url = generate_binder_url
    app.config.generate_colab_url = generate_colab_url
    app.config.get_check_example_api_reference = get_check_example_api_reference

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
