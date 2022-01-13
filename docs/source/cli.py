import typing as t
import sys
import types
import pathlib


CURRENT_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR.absolute())) # in case if project were not installed 


INDEX_FILE_TEMPLATE = """

Checks
======

.. py:module:: deepchecks.checks

.. automodule:: deepchecks.checks

.. currentmodule:: deepchecks.checks

.. autosummary::

    {checks_packages}

.. toctree::
    :caption: Checks
    :hidden:

    {checks_packages}

"""


CHECKS_FILE_TEMPLATE = """

{namespace}
============

.. py:module:: {module_fullname}

.. automodule:: {module_fullname}

Classes Summary
~~~~~~~~~~~~~~~

.. currentmodule:: {module_fullname}

.. autosummary::
    :recursive:
    :toctree: generated
    :template: check.rst

    {checks_names}

"""


def generate(
    output_dir: t.Optional[pathlib.Path] = None,
    dry_run: bool = False
):
    """
    Generate API References files for checks.

    Parameters
    ----------

    output_dir: Optional[Path], default None
        directory where to put prepared rst files
        in case of none, it will put prepared rst files into "__file__/api/checks/"
    
    dry_run: bool, default False
        flag to indicate that files content should be printed to the stdout
    
    """
    import deepchecks.checks
    from deepchecks.base import BaseCheck

    print("!! Checks API Reference Generation !!")
    print("!! Step 1. Collecting checks !!")

    checks_packages: t.Dict[str, types.ModuleType] = {
        getattr(deepchecks.checks, it).__name__: getattr(deepchecks.checks, it)
        for it in dir(deepchecks.checks)
        if isinstance(getattr(deepchecks.checks, it), types.ModuleType)
    }
    checks_packages_names = [ 
        fullname.split(".")[-1]
        for fullname in checks_packages.keys()
    ]

    index_file = INDEX_FILE_TEMPLATE.format(
        checks_packages="\n    ".join(checks_packages_names)
    )

    checks_files: t.Dict[str, str] = {} # Dict[file-name, file-content]

    for fullname, package in checks_packages.items():
        name = t.cast(str, fullname.split(".")[-1])
        checks = [
            getattr(package, it) 
            for it in dir(package) 
            if isinstance(getattr(package, it), type) and issubclass(getattr(package, it), BaseCheck)
        ]
        checks_names = [
            f"{it.__module__.replace(fullname+'.', '')}.{it.__name__}"
            for it in checks
        ]

        checks_files[name+".rst"] = CHECKS_FILE_TEMPLATE.format(
            namespace=name.capitalize(),
            module_fullname=fullname,
            checks_names="\n    ".join(checks_names)
        )
    
    if dry_run is True:
        print("!! Step 2. Printing Output !!")
        print("# ++++++ index.rst ++++++")
        print(index_file)
        
        for filename, content in checks_files.items():
            print(f"# ++++++ {filename} ++++++")
            print(content)
    
    else:
        print("!! Step 2. Writting Output !!")
        
        if output_dir is None:
            output_dir = CURRENT_DIR / "api" / "checks"

        if not output_dir.exists():
            raise RuntimeError(f"Output dir does not exist. {output_dir.absolute()}")

        for filename, content in checks_files.items():
            with (output_dir / filename).open("w") as f:
                f.write(content)
        
        with (output_dir / "index.rst").open("w") as f:
            f.write(index_file)


if __name__ == "__main__":
    _, cmd, *_ = sys.argv

    if cmd == "generate-checks-api":
        # writes rst files into CURRENT_DIR/api/checks
        generate()
    
    elif cmd == "dry-generate-checks-api":
        # prints rst files
        generate(dry_run=True)

    else:
        print(f"Unknow cmd - {cmd}")
        sys.exit(1)