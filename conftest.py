"""Put the repository root on ``sys.path`` so the tests run without installing.

``pip install -e .`` needs a PEP 660 backend and therefore a recent setuptools;
nothing here requires the package to be installed, so this removes the
dependency on that working.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
