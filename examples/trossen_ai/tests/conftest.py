"""Pytest config for the trossen_ai EE conversion tests."""
import sys
from pathlib import Path

# Put examples/trossen_ai on sys.path so `external.joint_to_ee...` imports resolve.
TROSSEN_AI_DIR = Path(__file__).resolve().parent.parent
if str(TROSSEN_AI_DIR) not in sys.path:
    sys.path.insert(0, str(TROSSEN_AI_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: needs the WXAI URDF + placo (lerobot RobotKinematics).",
    )
