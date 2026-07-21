# Start the Trossen Web App

This is the only startup guide for the two-machine setup:

- **HP Machine:** web app and model inference
- **Aloha Laptop:** robot and cameras

Run the following sections in order. Keep the physical E-stop within reach before
starting a robot session.

Use exactly one owner for each listening port. The HP Machine does not use
OpenPI systemd services in this deployment; its SSH tunnel and web app are
foreground commands. The Aloha Laptop uses systemd only for the robot gateway.

The normal startup model is deliberately mixed:

- **Aloha Laptop:** systemd keeps the robot gateway listening on `127.0.0.1:8001`.
- **HP Machine:** start the SSH tunnel manually in a foreground terminal.
- **HP Machine:** start the web app manually only after the tunnel health check
  succeeds.

## Port plan

| Machine | Listener | Expected owner | Purpose |
| --- | --- | --- | --- |
| Aloha Laptop | `127.0.0.1:8001` | Robot-gateway Uvicorn service | Robot gateway |
| Aloha Laptop | `127.0.0.1:8800` | SSH reverse-forward listener | Route gateway policy requests to the HP Machine |
| HP Machine | `127.0.0.1:8001` | One SSH tunnel process | Route the web app to the Aloha gateway |
| HP Machine | `127.0.0.1:8081` | One manual web-app Uvicorn process | Web UI and API |
| HP Machine | `127.0.0.1:8800` | Inference child, while a checkpoint is served | Model inference |

Using `8001` and `8800` on both machines is intentional: they are separate
loopback interfaces joined by SSH. In the gateway unit,
`OPENPI_GATEWAY_POLICY_PORT=8800` is an outbound destination; the gateway itself
only listens on the Uvicorn `--port 8001`.

## 1. Check the robot gateway service on the Aloha Laptop

This is always the first startup check on the Aloha Laptop:

```bash
systemctl --user status openpi-robot-gateway.service --no-pager
```

If the output says `Active: active (running)`, do not start another gateway.
Verify the existing gateway:

```bash
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
```

If status instead says `inactive`, `dead`, or `failed`—or if the health check
fails—restart the installed unit and immediately check both its status and
health endpoint:

```bash
systemctl --user restart openpi-robot-gateway.service
systemctl --user status openpi-robot-gateway.service --no-pager
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
```

Do not continue to the HP tunnel until status reports `active (running)` and the
health response reports `"ok":true` and `"gateway":"ready"`. If restart fails,
inspect the gateway log:

```bash
journalctl --user -u openpi-robot-gateway.service -n 100 --no-pager
```

If status says `Unit openpi-robot-gateway.service could not be found`, install
the Aloha-only unit from the `~/openpi-client-test` checkout:

```bash
mkdir -p ~/.config/systemd/user
cp ~/openpi-client-test/deploy/systemd/openpi-robot-gateway.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openpi-robot-gateway.service
systemctl --user status openpi-robot-gateway.service --no-pager
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
```

The tracked unit consistently uses `~/openpi-client-test` for both its working
directory and Python environment. If the Aloha checkout is stored elsewhere,
edit both paths in the copied unit before enabling it.

This does not mean the port runs literally forever. While
`openpi-robot-gateway.service` is active, it continuously listens on Aloha's
`127.0.0.1:8001`:

- **`enabled`:** starts automatically when the `edgeai` user's systemd manager
  starts.
- **`Restart=on-failure`:** systemd restarts the gateway after a crash.
- **Bound to `127.0.0.1`:** accessible only on the Aloha Laptop itself or through
  the SSH tunnel.

After a reboot, the user manager normally starts when `edgeai` logs in. Check
whether it also starts at boot and remains available after logout with:

```bash
loginctl show-user edgeai -p Linger
```

`Linger=yes` means the user manager can run without an interactive login. The
gateway still stops during shutdown or after an explicit
`systemctl --user stop`. It binds only to Aloha's loopback interface, so the HP
Machine reaches it through the SSH tunnel below.

## 2. Start the SSH tunnel manually on the HP Machine

Run this step on the HP Machine, not on the Aloha Laptop. The SSH tunnel is a
foreground process and is intentionally not managed by systemd for this setup.
One SSH connection provides both directions: HP port `8001` reaches the Aloha
gateway, and Aloha port `8800` reaches HP inference.

First check whether a working tunnel already owns HP port `8001`:

```bash
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
ss -ltnp 'sport = :8001'
pgrep -af 'ssh.*8001'
```

If the health check reports `"gateway":"ready"`, reuse that tunnel and skip the
next command. Otherwise, start one tunnel on the HP Machine:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8001:127.0.0.1:8001 \
  -R 127.0.0.1:8800:127.0.0.1:8800 \
  edgeai@192.168.50.219
```

Enter the SSH password when prompted and leave this terminal open. No output
after login is normal. If SSH reports `Address already in use`, another process
already owns HP port `8001`; inspect the commands above instead of starting a
second tunnel. In another HP terminal, verify the new local listener before
starting the web app:

```bash
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
```

The response must report `"ok":true` and `"gateway":"ready"`. Press `Ctrl+C`
in the SSH terminal to stop only the tunnel; the Aloha gateway service continues
running.

## 3. Start the web app manually on the HP Machine

Keep the SSH terminal from step 2 open. In a different HP terminal, change to
the Trossen directory and run this exact one-line command:

```bash
cd /home/saleha/openpi/examples/trossen_ai
OPENPI_CHECKPOINT_ROOT=/home OPENPI_REPO_ROOT=/home/saleha/openpi OPENPI_INFERENCE_PYTHON=/home/saleha/openpi/.venv/bin/python OPENPI_ROBOT_GATEWAY_URL=ws://127.0.0.1:8001/ws/robot .venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8081 --workers 1 --lifespan on
```

The values have distinct roles:

- `OPENPI_CHECKPOINT_ROOT=/home` selects the tree searched for model
  checkpoints.
- `OPENPI_REPO_ROOT=/home/saleha/openpi` identifies the OpenPI source checkout.
- `OPENPI_INFERENCE_PYTHON=/home/saleha/openpi/.venv/bin/python` selects the
  separate environment used for the inference child.
- `OPENPI_ROBOT_GATEWAY_URL=ws://127.0.0.1:8001/ws/robot` sends robot commands
  through the HP side of the SSH tunnel.
- The leading `.venv/bin/python` is the Trossen web-app environment, `8081` is
  the browser/API port, and `--workers 1` guarantees one process owns the model
  manager and robot-gateway client.

Leave this terminal open and open:

<http://127.0.0.1:8081>

If the HP Machine is accessed through remote VS Code, forward port `8081` in the
VS Code Ports panel and open the forwarded address.

Press `Ctrl+C` in this terminal to stop the web app after the robot session and
served checkpoint have both been stopped. Do not run a second manual Uvicorn
process; it can hold ports and controller leases.

## 4. Start a session

In the web app:

1. Select a checkpoint and click **Serve checkpoint**.
2. Wait until the checkpoint status is **ready**.
3. Select **Dry run**.
4. Click **Start Live**.

Dry run suppresses policy-driven arm commands, but it still connects to the real
arms and cameras. Only switch to autonomous mode after Dry run works correctly.

## Quick check

From the HP Machine, these commands should succeed:

```bash
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
curl -fsS --max-time 5 http://127.0.0.1:8081/api/health
```

The second response should report the robot gateway URL as
`ws://127.0.0.1:8001/ws/robot` and eventually show it as connected. If the
gateway health check succeeds but the web app reports another URL, stop the
manual web app with `Ctrl+C` and rerun the step 3 command with the exact
`OPENPI_ROBOT_GATEWAY_URL`.

## Diagnose `Address already in use`

Run these commands on the machine that printed the error:

```bash
sudo ss -ltnp | grep -E ':(8081|8001|8800)\b'
pgrep -af 'uvicorn|serve_policy|ssh.*(8001|8800)'
systemctl --user list-units --type=service --all | grep -i openpi
```

- On the Aloha Laptop, `8001` should be owned by the one robot-gateway service.
- On the HP Machine, `8001` should be owned by the one SSH tunnel and `8081` by
  one manual web-app Uvicorn process.
- HP port `8800` is normally absent until a checkpoint is being served.

If the expected owner is already healthy, there is nothing to fix: reuse or
restart it. Stop a stale process before starting its replacement. Do not change
ports merely to allow duplicate copies of the same component.

## Change a port

Systemd is not required to change ports. In the manual workflow, stop the
affected foreground process with `Ctrl+C`, change its command or environment,
and start it again. Keep every reference to a changed port consistent:

- **Web app (`8081`):** change `--port` in the manual step 3 command, then update
  the browser URL or VS Code forwarded port.
- **Robot gateway (`8001`):** change `--port` in the Aloha gateway unit, the HP
  SSH `-L` mapping, and `OPENPI_ROBOT_GATEWAY_URL` in the manual web-app command.
  Restart the gateway and recreate the SSH tunnel.
- **Inference (`8800`):** set the matching `OPENPI_INFERENCE_PORT` in the manual
  web-app command, `OPENPI_GATEWAY_POLICY_PORT` in the Aloha gateway unit, and
  both sides of the SSH `-R` mapping. Restart the gateway and web app, then
  recreate the SSH tunnel.

To verify that the expected ports are listening on either machine:

```bash
sudo ss -ltnp | grep -E ':(8081|8001|8800)\b'
```

## Stop everything

1. Click **Stop** in the web app and wait for the session to become idle.
2. Click **Stop serving** and wait for the model to stop.
3. Press `Ctrl+C` in the manually started HP web-app terminal.
4. Press `Ctrl+C` in the manually started SSH-tunnel terminal.
5. Stop the Aloha gateway only if the robot gateway should be shut down:
   `systemctl --user stop openpi-robot-gateway.service`.

The web E-stop is a software stop. Use the physical E-stop in an emergency.
