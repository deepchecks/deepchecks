# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import typing as t
import inspect
import os
import sys
import pathlib
import functools
from subprocess import check_output
# from sphinx.ext.autosummary import Autosummary
# from sphinx.ext.autosummary import get_documenter
# from docutils.parsers.rst import directives
# from sphinx.util.inspect import safe_getattr
# import re




# -- Project information -----------------------------------------------------

project = 'Deepchecks'
copyright = '2021, Deepchecks'
author = 'Deepchecks'
version = "TestVersoion" # This will be changed after merge!


CURRENT_DIR = pathlib.Path(__file__).parent
PROJECT_FIR = CURRENT_DIR.parent.parent


GIT = {
    "user": "deepchecks",
    "repo": "deepchecks",
    "documentation-path": "docs/source"
}


try:
    git_branch = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    GIT["branch"] = git_branch.decode().strip()
    tag = git_branch
    git_release = check_output(['git', 'describe', '--tags', '--always'])
    GIT["release"] = git_release.decode().strip()
except Exception as error:
    raise RuntimeError("Failed to extract commit hash!") from error


# -- General configuration ---------------------------------------------------

# Path to logo
#
html_logo = "./_static/deepchecks_logo.svg"


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
    
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'sphinx_search.extension',
]


html_baseurl = 'docs.deepchecks.com'
add_module_names = False
python_use_unqualified_type_names = True


# Autodoc settings
#
autodoc_class_signature = 'separated'

# Napoleon settings
#
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# Add any paths that contain templates here, relative to this directory.
#
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


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
            <a href="{{  env.config.generate_binder_url(env.docname) }}" style="vertical-align:text-bottom">
                <img
                    alt="Binder badge"
                    src="/_static/binder-badge.svg"
                    style="vertical-align:text-bottom">
            </a>
        </span>
        <span style="white-space: nowrap; margin-right: 10px;">
            <a href="{{  env.config.generate_colab_url(env.docname) }}" style="vertical-align:text-bottom">
                <img
                    alt="Colab badge"
                    src="/_static/colab-badge.svg"
                    style="vertical-align:text-bottom">
            </a>
        </span>
    </div>

"""


nbsphinx_epilog = r"""

{% set apipath =  env.config.get_check_example_api_reference(env.docname) %}

{% if apipath %}

.. rubric:: References
    
{{ apipath }}

{% endif %}

"""


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


#
html_sidebars = {
    "**": ["sidebar-nav-bs"]
}




# Theme specific options.
# See documentation of the 'pydata_sphinx_theme' theme
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html#adding-external-links-to-your-nav-bar
#
html_theme_options = {
    "show_nav_level": 4,
    "collapse_navigation": False,
    "navbar_end": ["search-field", "navbar-icon-links", "menu-dropdown",],
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


# -- Other -------------------------------------------------


def get_report_issue_url(sourcename: str) -> str:
    sourcename = sourcename.replace(".txt", "")
    return (
        f"https://github.com/{GIT['user']}/{GIT['repo']}/issues/new?"
        f"title=[Docs][Version:{version}] Issue within file: {sourcename}&"
        "labels=chore/documentation"
    )


def generate_colab_url(notebook_path: str) -> str:
    notebook_path = notebook_path.replace(".txt", "")
    notebook_path = notebook_path if notebook_path.endswith(".ipynb") else notebook_path + ".ipynb"
    notebook_name = notebook_path.split("/")[-1]
    
    if not is_example_notebook(notebook_name):
        raise RuntimeError(f"Not a notebook - {notebook_path}")
    
    template = (
        "https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{docpath}/{notebook_path}"
    )
    
    return template.format(
        user=GIT['user'],
        repo=GIT['repo'],
        branch=GIT['release'],
        docpath=GIT['documentation-path'],
        notebook_path=notebook_path
    )


def generate_binder_url(notebook_path: str) -> str:
    notebook_path = notebook_path.replace(".txt", "")
    notebook_path = notebook_path if notebook_path.endswith(".ipynb") else notebook_path + ".ipynb"
    notebook_name = notebook_path.split("/")[-1]
    
    if not is_example_notebook(notebook_name):
        raise RuntimeError(f"Not a notebook - {notebook_path}")

    template = (
        "https://mybinder.org/v2/gh/{user}/{repo}/{branch}?labpath={filepath}"
    )
    
    return template.format(
        user=GIT['user'],
        repo=GIT['repo'],
        branch=GIT['release'],
        filepath=f"{GIT['documentation-path']}/{notebook_path}"
    )


@functools.lru_cache(maxsize=None)
def get_example_notebooks() -> t.Tuple[pathlib.Path, ...]:
    examples_folder = CURRENT_DIR / "examples"

    if not examples_folder.exists() or not examples_folder.is_dir():
        raise RuntimeError("Did not find the folder with the example notebooks.")

    return tuple(it for it in examples_folder.glob("**/*.ipynb"))


@functools.lru_cache(maxsize=None)
def snake_case_to_camel_case(val: str) -> str:
    return "".join(it.capitalize() for it in val.split("_") if it)
    

def is_example_notebook(filepath: str) -> bool:
    notebook_name = filepath.split("/")[-1].replace(".txt", "")
    notebooks = {it.name for it in get_example_notebooks()}
    return notebook_name in notebooks


def get_check_example_api_reference(filepath: str) -> t.Optional[str]:

    if not filepath.startswith("examples/checks/"):
        return

    notebook_name = snake_case_to_camel_case(
        filepath.split("/")[-1]
        .replace(".txt", "")
        .replace(".ipynb", "")
        .replace(".py", "")
    )
    
    import deepchecks.checks
    check_clazz = getattr(deepchecks.checks, notebook_name, None)
    
    if check_clazz is None or not hasattr(check_clazz, "__module__"):
        return
    
    apipath = f"/api/checks/generated/{check_clazz.__module__}.{notebook_name}"
    result = f"* :doc:`{notebook_name} API <{apipath}>`"
    return result




# TODO: Not sure if we need this

# class AutoAutoSummary(Autosummary):

#     option_spec = {
#         'methods': directives.unchanged,
#         'attributes': directives.unchanged,
#         'classes': directives.unchanged
#     }

#     required_arguments = 1

#     @staticmethod
#     def get_members(obj, typ, include_public=None):
#         if not include_public:
#             include_public = []
#         items = []
#         for name in dir(obj):
#             try:
#                 documenter = get_documenter(safe_getattr(obj, name), obj)
#             except AttributeError:
#                 continue
#             if documenter.objtype == typ:
#                 items.append(name)
#         public = [x for x in items if x in include_public or not x.startswith('_')]
#         return public, items

#     def run(self):
#         clazz = str(self.arguments[0])
#         try:
#             (module_name, class_name) = clazz.rsplit('.', 1)
#             m = __import__(module_name, globals(), locals(), [class_name])
#             c = getattr(m, class_name)
#             if 'methods' in self.options:
#                 _, methods = self.get_members(c, 'method', ['__init__'])

#                 self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
#             if 'attributes' in self.options:
#                 _, attribs = self.get_members(c, 'attribute')
#                 self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
#             if 'classes' in self.options:
#                 _, classes = self.get_members(c, 'class')
#                 self.content = ["~%s.%s" % (clazz, cls) for cls in classes if not cls.startswith('_')]

#         finally:
#             return super(AutoAutoSummary, self).run()


# -- Registration of hooks ---------


def setup(app):
    # app.add_directive('autoautosummary', AutoAutoSummary)

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
