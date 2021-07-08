PYTHON_VERSION = 3.6.9

PIP = env/bin/pip
PYTHON = env/bin/python
VIRTUALENV = $(PYENV_ROOT)/versions/$(PYTHON_VERSION)/bin/virtualenv

env:
	pyenv install -s $(PYTHON_VERSION)  # make sure expected python version is available
	$(PYENV_ROOT)/versions/$(PYTHON_VERSION)/bin/pip install virtualenv  # ensure virtualenv is installed for given python
	if [ ! "$(shell $(PYTHON) -V)" = "Python $(PYTHON_VERSION)" ]; then \
		echo "WARNING: python version mismatch => reset virtualenv" && rm -rf env/; \
	fi
	if [ ! -d env/ ]; then \
		$(VIRTUALENV) env/; \
	fi
	$(PIP) install -qU pip

init:
	$(PIP) install -Ur requirements-dev.txt

clean: clean-build clean-pyc  ## remove all artifacts

clean-build:  ## remove build artifacts
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info

clean-pyc:  ## remove Python file artifacts
	-@find . -name '*.pyc' -not -path "./.tox/*" -follow -print0 | xargs -0 rm -f
	-@find . -name '*.pyo' -not -path "./.tox/*" -follow -print0 | xargs -0 rm -f
	-@find . -name '__pycache__' -type d -not -path "./.tox/*" -follow -print0 | xargs -0 rm -rf

test:
	env/bin/pytest

dist: clean
	#check-manifest
	$(PYTHON) setup.py sdist bdist_wheel
	ls -l dist
	env/bin/twine check dist/*

tag:
	git tag -a -m "Auto-generated tag" `sed -rn 's/^__version__\s*=\s*"(.+)"/\1/p' yolov5/__init__.py`
	git push --tags

release: clean dist tag
	env/bin/twine upload dist/* --repository-url https://pypi.psycle.dev/

lint:
	env/bin/black . --exclude '/(env)/'
	env/bin/pylint --disable=C yolov5
