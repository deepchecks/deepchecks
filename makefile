# This makefile helps with deepchecks Development environment
# including syntax checking, virtual environments creation, 
# test running and coverage
# This Makefile is based on Makefile by jidn: https://github.com/jidn/python-Makefile/blob/master/Makefile

# Package = Source code Directory
PACKAGE = deepchecks

# Requirements file
REQUIRE = requirements.txt

# python3 binary takes predecence over python binary,
# this variable is used when setting python variable, (Line 18)
# and on 'env' goal ONLY 
# If your python path binary name is not python/python3,
# override using ext_python=XXX and it'll propogate into python variable, too
ext_py := $(shell which python3 || which python)

# Override by putting in commandline python=XXX when needed.
python = $(shell echo ${ext_py} | rev | cut -d '/' -f 1 | rev)
TESTDIR = tests
ENV = venv
repo = pypi

# System Envs
BIN := $(ENV)/bin
pythonpath= PYTHONPATH=.

# Venv Executables
PIP := $(BIN)/pip
PYTHON := $(BIN)/$(python)
ANALIZE := $(BIN)/pylint
COVERAGE := $(BIN)/coverage
TEST_RUNNER := $(BIN)/pytest
TOX := $(BIN)/tox
TWINE := $(BIN)/twine
APIDOC := $(BIN)/sphinx-apidoc
SPHINX_BUILD := $(BIN)/sphinx-build
JUPYTER := $(BIN)/jupyter

# Project Settings
PKGDIR := $(or $(PACKAGE), ./)
SOURCES := $(or $(PACKAGE), $(wildcard *.py))

# Installation packages
INSTALLATION_PKGS = wheel setuptools

REQUIREMENTS := $(shell find . -name $(REQUIRE))
REQUIREMENTS_LOG := .requirements.log

# Test and Analyize
ANALIZE_PKGS = pylint pydocstyle 
TEST_CODE := tests/
TEST_RUNNER_PKGS = pytest pytest-cov pyhamcrest nbval
NOTEBOOK_DIR = ./notebooks/examples ./notebooks/checks
NOTEBOOK_SANITIZER_FILE= ./notebooks/.nbval-sanitizer

PYLINT_LOG = .pylint.log

# Coverage vars
COVERAGE_LOG = .cover.log
COVERAGE_FILE = default.coveragerc
COVERAGE_RC := $(wildcard $(COVERAGE_FILE))
COVER_ARG := --cov-report term-missing --cov=$(PKGDIR) \
	$(if $(COVERAGE_RC), --cov-config $(COVERAGE_RC))

# Sphinx
SPHINX_PKGS = sphinx sphinx_rtd_theme sphinx-markdown-builder


EGG_INFO := $(subst -,_,$(PROJECT)).egg-info
EGG_LINK = venv/lib/python3.7/site-packages/deepchecks.egg-link

### Main Targets ######################################################

.PHONY: help env all 

help:
	@echo "env      -  Create virtual environment and install requirements"
	@echo "               python=PYTHON_EXE   interpreter to use, default=python,"
	@echo "						    	when creating new env and python binary is 2.X, use 'make env python=python3' \n"
	@echo "validate - Run style checks 'pylint' , 'docstring' and 'notebook'"
	@echo "		pylint docstring notebook -   sub commands of validate \n"
	@echo "test -      TEST_RUNNER on '$(TESTDIR)'"
	@echo "            args=\"<pytest Arguements>\"  optional arguments"
	@echo "coverage -  Get coverage information, optional 'args' like test\n"
	@echo "jupyter - Deploy jupyer-notebook using './notebook' directory"
	@echo "					 args=\"<jupyter Arguments\" -passable\n"
	@echo "tox      -  Test against multiple versions of python as defined in tox.ini"
	@echo "clean | clean-all -  Clean up | clean up & removing virtualenv"

all: validate test notebook


env: $(REQUIREMENTS_LOG)
$(PIP):
	$(info #### Remember to source new environment  [ $(ENV) ] ####)
	@echo "external python_exe is $(ext_py)"
	test -d $(ENV) || $(ext_py) -m venv $(ENV) 
$(REQUIREMENTS_LOG): $(PIP) $(REQUIREMENTS)
	$(PIP) install --upgrade pip
	$(PIP) install $(INSTALLATION_PKGS)
	for f in $(REQUIREMENTS); do \
	  $(PIP) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done



### Static Analysis ######################################################

.PHONY: validate pylint docstring

validate: $(REQUIREMENTS_LOG) pylint docstring

pylint: $(ANALIZE)
	$(ANALIZE) $(SOURCES) $(TEST_CODE)
docstring: $(ANALIZE) # We Use PEP257 Style Python Docstring
	$(PYTHON) -m pydocstyle --convention=pep257 --add-ignore=D107 $(SOURCES)

$(ANALIZE): $(PIP)
	$(PIP) install --upgrade $(ANALIZE_PKGS) | tee -a $(REQUIREMENTS_LOG)


### Testing ######################################################

.PHONY: test coverage notebook

test: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
	$(pythonpath) $(TEST_RUNNER) $(args) $(TESTDIR)


notebook: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
# if deepchecks is not installed, we need to install it for testing porpuses,
# as the only time you'll need to run make is in dev mode, we're installing
# deepchecks in development mode
	$(PIP) install --no-deps -e .
	$(pythonpath) $(TEST_RUNNER) --nbval $(NOTEBOOK_DIR) --sanitize-with $(NOTEBOOK_SANITIZER_FILE)

$(TEST_RUNNER):
	$(PIP) install $(TEST_RUNNER_PKGS) | tee -a $(REQUIREMENTS_LOG)

coverage: $(REQUIREMENTS_LOG) $(TEST_RUNNER) 
	$(pythonpath) $(TEST_RUNNER) $(args) $(COVER_ARG) $(TESTDIR) | tee -a $(COVERAGE_LOG)


# This is Here For Legacy || future use case,
# our PKGDIR is in its own directory so we dont really need to remove the ENV dir.
$(COVERAGE_FILE):
ifeq ($(PKGDIR),./)
ifeq (,$(COVERAGE_RC))
	# If PKGDIR is root directory, ie code is not in its own directory
	# then you should use a .coveragerc file to remove the ENV directory
	$(info Rerun make to discover autocreated $(COVERAGE_FILE))
	@echo -e "[run]\nomit=$(ENV)/*" > $(COVERAGE_FILE)
	@cat $(COVERAGE_FILE)
	@exit 68
endif
endif


# tox checks for all python versions matrix
tox: $(TOX)
	$(TOX)

$(TOX): $(PIP)
	$(PIP) install tox | tee -a $(REQUIREMENTS_LOG)


### Cleanup ######################################################

.PHONY: clean clean-env clean-all clean-build clean-test clean-dist clean-docs

clean: clean-dist clean-test clean-build clean-docs

clean-env: clean
	-@rm -rf $(ENV)
	-@rm -rf $(REQUIREMENTS_LOG)
	-@rm -rf $(COVERAGE_LOG)
	-@rm -rf $(PYLINT_LOG)
	-@rm -rf .tox

clean-all: clean clean-env

clean-build:
	@find $(PKGDIR) -name '*.pyc' -delete
	@find $(PKGDIR) -name '__pycache__' -delete
	@find $(TESTDIR) -name '*.pyc' -delete 2>/dev/null || true
	@find $(TESTDIR) -name '__pycache__' -delete 2>/dev/null || true
	-@rm -rf $(EGG_INFO)
	-@rm -rf __pycache__

clean-test:
	-@rm -rf .pytest_cache
	-@rm -rf .coverage

clean-dist:
	-@rm -rf dist build

clean-docs:
	-@rm -rf docs/_build
	-@rm -rf docs/deepcheckss

### Release ######################################################
.PHONY: authors register dist upload .git-no-changes ammend release

authors:
	echo "Authors\n=======\n\nA huge thanks to all of our contributors:\n\n" > AUTHORS.md
	git log --raw | grep "^Author: " | cut -d ' ' -f2- | cut -d '<' -f1 | sed 's/^/- /' | sort | uniq >> AUTHORS.md

dist: test
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel

# upload expects to get all twine args as environment,
# refer to https://twine.readthedocs.io/en/latest/ for more information
upload: $(TWINE) 
	$(TWINE) upload dist/*

ammend:
	git add deepchecks/version.py
	git commit --amend --no-edit


.git-no-changes:
	@if git diff --name-only --exit-code;       \
	then                                        \
		echo Git working copy is clean...;        \
	else                                        \
		echo ERROR: Git working copy is dirty!;   \
		echo Commit your changes and try again.;  \
		exit -1;                                  \
	fi;

release: version dist upload


$(TWINE): $(PIP)
	$(PIP) install twine

#if version variable is passed, the release version will be modified to this version.
version: 
ifeq ($(version),)
else
	@sed -i -E "s/__version__\ +=\ +'.*+'/__version__ = '${version}'/g" deepchecks/version.py
endif


### Documentation
.PHONY: docs website dev-docs gen-static-notebooks

API_REFERENCE_DIR=api-reference
WEBSITE_DIR=docs/_website
DOCOSAURUS := docs/_website/node_modules/.bin/docusaurus-start

$(DOCOSAURUS):
	@cd $(WEBSITE_DIR) ; \
	npm install

$(APIDOC): env
	$(PIP) install $(SPHINX_PKGS)	

gen-static-notebooks: $(JUPYTER)
	 $(BIN)/jupyter nbconvert --to html --output-dir $(WEBSITE_DIR)/static/notebooks/  ./notebooks/*/*/*.ipynb 

docs: $(APIDOC)
	$(pythonpath) $(BIN)/sphinx-apidoc -t docs/_templates -f ./deepchecks -o docs/$(API_REFERENCE_DIR)
	$(pythonpath) $(BIN)/sphinx-build -M markdown docs docs/_build/
	@rm -rf docs/api-reference
	@find docs/_build/markdown/ -name '*.md' | xargs sed '/^$$/N;/^\n$$/D'  -i

website: docs gen-static-notebooks
	@rm -rf $(WEBSITE_DIR)/docs/$(API_REFERENCE_DIR)
	@cp -rf docs/_build/markdown/$(API_REFERENCE_DIR) $(WEBSITE_DIR)/docs/$(API_REFERENCE_DIR)/
	@rm -rf docs/_build/markdown



dev-docs: $(DOCOSAURUS) website
	@cd $(WEBSITE_DIR) &&  \
	npm start





### System Installation ######################################################
.PHONY: develop install download jupyter

develop:
	$(PYTHON) setup.py develop

install: 
	$(PYTHON) setup.py install

download:
	$(PIP) install $(PROJECT)

jupyter: $(JUPYTER)
	$(BIN)/jupyter-notebook $(args) --notebook-dir=$(NOTEBOOK_DIR)

$(JUPYTER):
	$(PIP) install jupyter
