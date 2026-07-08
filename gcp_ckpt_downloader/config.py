from __future__ import annotations

import os

DEFAULT_BUCKET = os.environ.get("GCP_CKPT_BUCKET", "gs://tii-aiccu-falcon-vla-europe-west1/")
DEFAULT_DEST = os.environ.get("GCP_CKPT_DEST", "/home/ibrahim/storage/VLA_MODELS/")
GCP_PROJECT = os.environ.get("GCP_CKPT_PROJECT", "falcon-training-gpu")
