# Makefile for Sphinx documentation
#

ENV = ../venv
# System Envs
BIN := $(ENV)/bin
pythonpath= PYTHONPATH=..

SPHINXBUILDCMD=$(BIN)/sphinx-build

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = $(pythonpath) $(SPHINXBUILDCMD)
PAPER         =
BUILDDIR      = build
SOURCEDIR     = ./source

# Do not use local Django settings during the docs build
export DJANGO_SETTINGS_SKIP_LOCAL = True

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILDCMD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SOURCEDIR)

.PHONY: help clean html show

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  show       to open the static html files in local web browser"
	@echo "  clean      clean build directory"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."


show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/index.html')"
