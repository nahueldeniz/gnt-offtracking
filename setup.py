"""Shim for pip older than 21.3, which cannot do a PEP 660 editable install.

The metadata lives in pyproject.toml; this file only lets `pip install -e .`
work on an older pip.  The studies do not need the package to be installed:
the runners put the repository root on PYTHONPATH.
"""

from setuptools import setup

setup()
