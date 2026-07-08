# GCP Checkpoint Downloader

A web application for browsing and downloading PyTorch model checkpoints from Google Cloud Storage.

## Setup

### One-time authentication

Enable downloads via Application Default Credentials (ADC):

```bash
gcloud auth login --update-adc
gcloud config set project falcon-training-gpu
gcloud auth list           # confirm your account is ACTIVE
```

## Running the Application

### From the repository

```bash
uv run python -m gcp_ckpt_downloader --port 8070
```

### After installation

If you've installed the package (e.g., `pip install .` or `uv pip install -e .`), use:

```bash
gcp-ckpt-downloader --port 8070
```

Then open your browser to `http://<inference-machine>:8070` to:
1. Browse available checkpoint folders in the GCS bucket
2. Select checkpoints to download
3. Confirm or adjust the destination path (default: `/home/ibrahim/storage/VLA_MODELS/`)
4. Click **Download** and watch progress

## Command-line Options

- `--host`: Server bind address (default: `0.0.0.0`)
- `--port`: Server bind port (default: `8070`)

Example:

```bash
gcp-ckpt-downloader --host 127.0.0.1 --port 9000
```
