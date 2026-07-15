"""Pytest config for webapp unit tests.

Adds the webapp directory to sys.path so test modules can do bare imports
(e.g. ``import runners``, ``import session``) without package qualification.
Also adds the parent examples/trossen_ai directory so that ``import
robot_control`` works (robot_control.py lives there, not inside webapp/).
"""
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

TROSSEN_AI_DIR = WEBAPP_DIR.parent
if str(TROSSEN_AI_DIR) not in sys.path:
    sys.path.insert(0, str(TROSSEN_AI_DIR))
