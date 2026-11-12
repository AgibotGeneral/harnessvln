PYTHON ?= python

.PHONY: check
check:
	$(PYTHON) tools/check_release.py
