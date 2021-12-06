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
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath('../deepchecks'))

# -- Project information -----------------------------------------------------

project = 'deepchecks'
copyright = '2021, deepchecks'
author = 'deepchecks'
tag = 'main'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'autoapi.extension'
]

autoapi_type = 'python'
autoapi_dirs = ['../deepchecks']
autoapi_options = ['members', 'undoc-members', 'show-module-summary', 'imported-members', 'inherited-members']
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_template_dir = "_templates/autoapi"


add_module_names = False
python_use_unqualified_type_names = True

# Autodoc settings
autodoc_class_signature = 'separated'

# Napoleon settings
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
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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


# note: cupy_backends is excluded as it is undocumented
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
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
