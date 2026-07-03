# Real-robot inference for ACT (openpi-compatible)

gcp: gcloud storage ls gs://tii-aiccu-falcon-vla-europe-west1/june_2026_batch2/falcon-act/act_aloha_14_25_kl10/
/home/ibrahim/storage/VLA_MODELS/april_2026_batch3/act
This serves a trained ACT checkpoint to the **stock openpi robot client**
(`examples/trossen_ai/main.py`) over websocket. The robot client is unchanged in
spirit — it just points at this server instead of a pi0 server. ACT is a
visuomotor policy, so the `--task_prompt` is accepted and ignored.

```
robot box (openpi env)                      GPU box (this act repo)
┌────────────────────────────┐  ws :8000   ┌──────────────────────────────────┐
│ examples/trossen_ai/main.py│ ──────────► │ python -m inference.serve_policy │
│  WebsocketClientPolicy     │ ◄────────── │  ActPolicy → model.predict_action│
│  obs{state,images,prompt}  │  actions    │  returns {"actions": (25,14)}    │
└────────────────────────────┘             └──────────────────────────────────┘
```

## Best checkpoints (HF snapshots — point `--hf_ckpt` at one of these)

| kl_weight | episodic MSE | HF export dir |
|-----------|--------------|----------------|
| **10 (best)** | 0.00025 | `…/act_aloha_14_25_kl10/hf_export_best` |
| 0.1 | 0.00028 | `…/act_aloha_14_25_kl0p1/hf_export_best` |
| 1 | 0.00040 | `…/act_aloha_14_25_kl1/hf_export_best` |

Full base path: `/lustre1/tier2/projects/falcon-vla/vla_training/personal_temp/manisha/`
Raw checkpoints are `…/policy_best.ckpt` in the same dirs (re-export with
`scripts/convert_hf/convert_act_to_hf.py` if needed).

## 1. Start the server (GPU box, this repo's env)

The server needs `websockets` + `msgpack` (not in the `aloha` env yet):

```bash
conda activate aloha
pip install "websockets>=13" msgpack

cd /lustre1/tier2/users/manisha.lingala/act
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m inference.serve_policy \
    --hf_ckpt /lustre1/tier2/projects/falcon-vla/vla_training/personal_temp/manisha/act_aloha_14_25_kl10/hf_export_best \
    --host 0.0.0.0 --port 8000 \
    --camera_map primary=cam_high,secondary=cam_low,wrist=cam_right_wrist
```

Liveness check from anywhere: `curl http://<SERVER_IP>:8000/healthz` → `OK`.

## 2. Point the robot client at the server (openpi repo, robot box)

No new files needed in openpi — just run the existing example against this server:

```bash
# in your openpi clone, on the robot
python examples/trossen_ai/main.py \
    --policy_host <SERVER_IP> --policy_port 8000 \
    --mode test \
    --task_prompt "pick the cup and place it in the orange basket"
```

`--mode test` logs actions WITHOUT moving the arm — always do this first.
Switch to `--mode autonomous` only once the logged actions look sane.

The client already sends the exact contract this server expects
(`examples/trossen_ai/main.py`, ~line 177):
```python
observation = {"state": joint_positions, "images": {cam: chw_img}, "prompt": task_prompt}
response   = self.policy_client.infer(observation)   # -> {"actions": (25, 14)}
```
The client's `action_chunk_size = 25` and `rate_of_inference = 25` already match
this policy's `chunk_size = 25`. Leave them as-is.

## 3. THREE things you MUST verify before `--mode autonomous`

These are deployment-correctness issues the code cannot check for you:

1. **Joint order.** The client builds `state` from the robot's `.pos` keys and
   applies `actions` in `robot._joint_ft` order. This must equal the 14-dim order
   of the **RLDS training dataset** (`aidrc_cups_manipulation_14`). If the robot
   enumerates joints differently, every prediction is a scrambled pose. Print both
   orders once and confirm they line up.

2. **Camera map.** `--camera_map model_cam=robot_cam`. The model expects
   `primary, secondary, wrist`; map each to the physical camera that fed that slot
   **during data collection**. The default assumes `secondary=cam_low`, but your
   client config has `cam_low` commented out (only `cam_high`, `cam_left_wrist`,
   `cam_right_wrist` are enabled). Either re-enable `cam_low` on the robot, or
   remap (e.g. `secondary=cam_left_wrist`) to match how you trained — guessing
   wrong here degrades the policy badly.

3. **Image resolution.** The client downsamples to `224×224` (a pi0 requirement)
   at `main.py` line ~171: `cv2.resize(image_hwc, (224, 224))`. ACT was trained at
   `480×640`. The server upscales back to `480×640`, but you lose detail. For best
   results change that line to `cv2.resize(image_hwc, (640, 480))` (W,H) so the
   client sends full resolution. The server's `--image_height/--image_width`
   default to `480×640`; keep them matched to training.

## Notes

- **No language.** Different cups/goals must be distinguishable from the camera
  view; ACT cannot follow `--task_prompt`. If two tasks share an identical first
  frame, ACT will average them — use the language-conditioned VLA for that.
- **Wire format.** `inference/msgpack_numpy.py` and `inference/server.py` are
  vendored from openpi so the server is byte-compatible with `openpi_client`
  without installing the full `openpi` package.
- **First-step jump.** The client already smoothly interpolates to the first
  predicted pose (`move_to_start_position`) to avoid a velocity-limit trip.

##### testing 

  cd ~/openpi/act
export PYTHONPATH="$PWD:$PYTHONPATH"
CKPT=/home/ibrahim/storage/VLA_MODELS/april_2026_batch3/act/hf_export_best

python -m inference.serve_policy \
  --hf_ckpt "$CKPT" \
  --host 0.0.0.0 --port 8000 \
  --camera_map primary=cam_high,secondary=cam_low,wrist=cam_right_wrist

#### test mode 
python examples/trossen_ai/main.py \
  --policy_host localhost --policy_port 8000 \
  --mode test \
  --task_prompt "pick the cup and place it in the orange basket"


######## autonomous mode

python examples/trossen_ai/main.py \
  --policy_host localhost --policy_port 8000 \
  --mode autonomous \
  --task_prompt "pick the cup and place it in the orange basket"


##################### error
lerobot-newest) (base) ibrahim@HP-Z8-G4-Workstation:~/openpi/act$  cd /home/ibrahim/openpi/act ; /usr/bin/env /home/ibrahim/openpi/lerobot-newest/.venv/bin/python /home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 56875 -- /home/ibrahim/openpi/act/serve_policy.py --hf_ckpt /home/ibrahim/storage/VLA_MODELS/april_2026_batch3/act/hf_export_best --camera_map primary=cam_high,secondary=cam_right_wrist,wrist=cam_left_wrist --image_height 480 --image_width 640 --port 8801 
Encountered exception while importing policy: No module named 'policy'
Traceback (most recent call last):
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 71, in <module>
    cli.main()
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 542, in main
    run()
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 361, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 310, in run_path
    return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 127, in _run_module_code
    _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
  File "/home/ibrahim/.vscode/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 118, in _run_code
    exec(code, run_globals)
  File "/home/ibrahim/openpi/act/serve_policy.py", line 107, in <module>
    main()
  File "/home/ibrahim/openpi/act/serve_policy.py", line 79, in main
    model = load_model(args.hf_ckpt, device)
  File "/home/ibrahim/openpi/act/serve_policy.py", line 38, in load_model
    model = AutoModel.from_pretrained(hf_ckpt_dir, trust_remote_code=True)
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/site-packages/transformers/models/auto/auto_factory.py", line 553, in from_pretrained
    model_class = get_class_from_dynamic_module(
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/site-packages/transformers/dynamic_module_utils.py", line 540, in get_class_from_dynamic_module
    final_module = get_cached_module_file(
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/site-packages/transformers/dynamic_module_utils.py", line 365, in get_cached_module_file
    modules_needed = check_imports(resolved_module_file)
  File "/home/ibrahim/miniconda3/envs/act_infer/lib/python3.8/site-packages/transformers/dynamic_module_utils.py", line 197, in check_imports
    raise ImportError(
ImportError: This modeling file requires the following packages that were not found in your environment: policy. Run `pip install policy`
(lerobot-newest) (base) ibrahim@HP-Z8-G4-Workstation:~/openpi/act$ 