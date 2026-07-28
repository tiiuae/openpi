"""Apply the saved settings in `realsense_settings.json` to the RealSense cameras.

The client reads its frames from v4l2loopback nodes (`/dev/video40-42`) fed by gstreamer, and
loopback devices expose no camera controls, so the settings have to go to the physical cameras
through librealsense. The JSON is keyed by the serial librealsense reports (the ASIC serial,
e.g. `230322270230`), not the USB serial under `/dev/v4l/by-id/`.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Writing a manual value clears the matching auto flag (setting `white_balance` turns
# `enable_auto_white_balance` off), so the auto flags have to be applied last.
_AUTO_OPTIONS = ("enable_auto_exposure", "enable_auto_white_balance")


def apply_realsense_settings(settings_path: str | Path) -> None:
    """Push the settings for each listed serial onto that camera, if it is connected."""
    import pyrealsense2 as rs

    settings_by_serial = json.loads(Path(settings_path).read_text())
    devices = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}

    for serial, settings in settings_by_serial.items():
        device = devices.get(serial)
        if device is None:
            logger.warning("RealSense %s is not connected, skipping its settings", serial)
            continue

        # "Enable Auto Exposure" -> "enable_auto_exposure", auto flags last.
        values = {name.lower().replace(" ", "_"): float(v) for name, v in settings.items()}
        ordered = sorted(values.items(), key=lambda kv: kv[0] in _AUTO_OPTIONS)

        for sensor in device.query_sensors():
            options = {str(opt.name): opt for opt in sensor.get_supported_options()}
            for name, value in ordered:
                if name not in options:
                    continue  # not an option of this sensor
                try:
                    sensor.set_option(options[name], value)
                except RuntimeError as e:
                    logger.warning("RealSense %s: could not set %s=%g: %s", serial, name, value, e)
        logger.info("Applied %s settings to RealSense %s", len(values), serial)
