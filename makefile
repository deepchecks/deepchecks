# This makefile helps with mlchecks Development environment
# including syntax checking, virtual environments creation, 
# test running and coverage
# This Makefile is based on Makefile by jidn: https://github.com/jidn/python-Makefile/blob/master/Makefile

# Package = Source code Directory
PACKAGE = mlchecks

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

# System Envs
BIN := $(ENV)/bin

# Venv Executables
PIP := $(BIN)/pip
PYTHON := $(BIN)/$(python)
ANALIZE := $(BIN)/pylint
COVERAGE := $(BIN)/coverage
TEST_RUNNER := $(BIN)/py.test
TOX := $(BIN)/tox

# Project Settings
PKGDIR := $(or $(PACKAGE), ./)
SOURCES := $(or $(PACKAGE), $(wildcard *.py))

INSTALLATION_PKGS = wheel setuptools

REQUIREMENTS := $(shell find . -name $(REQUIRE))
REQUIREMENTS_LOG := .requirements.log

ANALIZE_PKGS = pylint pydocstyle
TEST_CODE := $(wildcard $(TESTDIR)/*.py)
TEST_RUNNER_PKGS = pytest pyhamcrest

COVERAGE_FILE = default.coveragerc
COVERAGE_PKGS = pytest-cov
COVERAGE_RC := $(wildcard $(COVERAGE_FILE))
COVER_ARG := --cov-report term-missing --cov=$(PKGDIR) \
	$(if $(COVERAGE_RC), --cov-config $(COVERAGE_RC))

EGG_INFO := $(subst -,_,$(PROJECT)).egg-info

### Main Targets ######################################################

.PHONY: help env all ci activate

help:
	@echo "env        Create virtual environment and install requirements"
	@echo "             python=PYTHON_EXE   interpreter to use, default=python,"
	@echo "							when creating new env and python binary is 2.X, use 'make env python=python3'"
	@echo "validate      Run style checks 'pylint' and 'docstring'"
	@echo "test       TEST_RUNNER on '$(TESTDIR)'"
	@echo "             args=\"-x --pdb --ff\"  optional arguments"
	@echo "coverage   Get coverage information, optional 'args' like test"
	@echo "tox        Test against multiple versions of python"
	@echo "upload     Upload package to PyPI"
	@echo "clean clean-all  Clean up and clean up removing virtualenv"

all: validate test

# CI is same as all, but they may be different in the future so we'll have them both
ci: validate test


env: $(REQUIREMENTS_LOG)
$(PIP):
	$(info #### Remember to source new environment  [ $(ENV) ] ####)
	@echo "external python_exe is $(ext_py)"
	test -d $(ENV) || $(ext_py) -m venv $(ENV) 
$(REQUIREMENTS_LOG): $(PIP) $(REQUIREMENTS)
	$(PIP) install $(INSTALLATION_PKGS)
	for f in $(REQUIREMENTS); do \
	  $(PIP) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done



### Static Analysis ######################################################

.PHONY: validate pylint docstring

validate: $(REQUIREMENTS_LOG) pylint docstring

pylint: $(ANALIZE)
	$(ANALIZE) $(SOURCES) $(TEST_CODE)
docstring: $(ANALIZE) # We Use Google Style Python Docstring
	$(PYTHON) -m pydocstyle $(SOURCES) $(TEST_CODE)

$(ANALIZE):
	$(PIP) install --upgrade $(ANALIZE_PKGS) | tee -a $(REQUIREMENTS_LOG)


### Testing ######################################################

.PHONY: test coverage

test: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
	$(TEST_RUNNER) $(args) $(TESTDIR)

$(TEST_RUNNER):
	$(PIP) install $(TEST_RUNNER_PKGS) | tee -a $(REQUIREMENTS_LOG)

coverage: $(REQUIREMENTS_LOG) $(COVERAGE) $(COVERAGE_FILE)
	$(TEST_RUNNER) $(args) $(COVER_ARG) $(TESTDIR)


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

$(COVERAGE): $(PIP)
	$(PIP) install $(COVERAGE_PKGS) | tee -a $(REQUIREMENTS_LOG)

# tox checks for all python versions matrix
tox: $(TOX)
	$(TOX)

$(TOX): $(PIP)
	$(PIP) install tox | tee -a $(REQUIREMENTS_LOG)


### Cleanup ######################################################

.PHONY: clean clean-env clean-all clean-build clean-test clean-dist

.PHONY: clean clean-env clean-all clean-build clean-test clean-dist

clean: clean-dist clean-test clean-build

clean-env: clean
	-@rm -rf $(ENV)
	-@rm -rf $(REQUIREMENTS_LOG)
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


### Release ######################################################
.PHONY: authors register dist upload .git-no-changes

authors:
	echo "Authors\n=======\n\nA huge thanks to all of our contributors:\n\n" > AUTHORS.md
	git log --raw | grep "^Author: " | cut -d ' ' -f2- | cut -d '<' -f1 | sed 's/^/- /' | sort | uniq >> AUTHORS.md

register:
	$(PYTHON) setup.py register -r pypi

dist: test
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel

upload: .git-no-changes register
	$(PYTHON) setup.py sdist upload -r pypi
	$(PYTHON) setup.py bdist_wheel upload -r pypi

.git-no-changes:
	@if git diff --name-only --exit-code;       \
	then                                        \
		echo Git working copy is clean...;        \
	else                                        \
		echo ERROR: Git working copy is dirty!;   \
		echo Commit your changes and try again.;  \
		exit -1;                                  \
	fi;


### System Installation ######################################################
.PHONY: develop install download

develop:
	$(PYTHON) setup.py develop

install:
	$(PYTHON) setup.py install

download:
	$(PIP) install $(PROJECT)