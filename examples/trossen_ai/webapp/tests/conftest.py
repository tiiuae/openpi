"""Pytest config for webapp unit tests.

Adds the webapp directory to sys.path so test modules can do bare imports
(e.g. ``import runners``, ``import session``) without package qualification.
The parent examples/trossen_ai directory is already on sys.path via pytest's
package-traversal, so ``from policy_connect import ...`` works too.
"""
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
