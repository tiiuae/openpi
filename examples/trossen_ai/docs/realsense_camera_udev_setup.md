# Persistent RealSense Camera Symlinks via udev

A guide to permanently fix Intel RealSense D405 camera device paths on Linux,
so `/dev/cam_high`, `/dev/cam_wrist_right`, and `/dev/cam_wrist_left` always
point to the correct physical cameras regardless of boot order or re-plugging.

---

## Background

Linux assigns `/dev/videoX` numbers dynamically at boot. Three identical RealSense
cameras will each enumerate 6 video nodes, and their numbers can shift between
reboots. The fix is a **udev rule** that creates stable symlinks based on each
camera's burned-in serial number.

---

## Step 1 — Find Each Camera's Serial

Run this to list the serial for every video device:

```bash
for dev in /dev/video*; do
  echo "=== $dev ==="
  udevadm info --query=all --name=$dev | grep -E 'ID_SERIAL|ID_VENDOR_ID|ID_MODEL_ID'
  echo ""
done
```

You'll see each RealSense D405 has a long `ID_SERIAL` string like:

```
E: ID_SERIAL=Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_418643070428
E: ID_SERIAL_SHORT=418643070428
```

Group the nodes by `ID_SERIAL_SHORT` — each camera occupies 6 consecutive nodes.

---

## Step 2 — Identify the RGB Node Index

Each RealSense exposes 6 video nodes per camera (depth, IR left, IR right, color, etc.).
You need the index of the **color/RGB stream** within that group.

Check the index of each node:

```bash
for dev in /dev/video0 /dev/video1 /dev/video2 /dev/video3 /dev/video4 /dev/video5; do
  echo -n "$dev → index: "
  cat /sys/class/video4linux/$(basename $dev)/index
done
```

Then confirm visually which node is RGB:

```bash
ffplay /dev/video0
ffplay /dev/video4   # etc.
```

> **On this machine the RGB stream is at `index 4` for all three cameras.**

---

## Step 3 — Write the udev Rules

```bash
sudo nano /etc/udev/rules.d/99-cameras.rules
```

Paste the following (replace serials if cameras change):

```
# cam_high RGB — serial 418643070349
SUBSYSTEM=="video4linux", \
  ENV{ID_SERIAL}=="Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_418643070349", \
  ATTR{index}=="4", \
  SYMLINK+="cam_high"

# cam_wrist_right RGB — serial 418643070357
SUBSYSTEM=="video4linux", \
  ENV{ID_SERIAL}=="Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_418643070357", \
  ATTR{index}=="4", \
  SYMLINK+="cam_wrist_right"

# cam_wrist_left RGB — serial 418643070428
SUBSYSTEM=="video4linux", \
  ENV{ID_SERIAL}=="Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_418643070428", \
  ATTR{index}=="4", \
  SYMLINK+="cam_wrist_left"
```

---

## Step 4 — Reload and Verify

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger

ls -la /dev/cam_*
```

Expected output:

```
lrwxrwxrwx ... /dev/cam_high        -> video16
lrwxrwxrwx ... /dev/cam_wrist_right -> video10
lrwxrwxrwx ... /dev/cam_wrist_left  -> video4
```

> The `videoX` numbers may differ on a new machine — that's fine.
> What matters is that `cam_high` always points to serial `418643070349`, etc.

---

## Camera Serial Reference

| Symlink | Role | Serial Short | Full ID_SERIAL |
|---|---|---|---|
| `/dev/cam_high` | Overhead camera | `418643070349` | `Intel_R__RealSense_TM__Depth_Camera_405_..._418643070349` |
| `/dev/cam_wrist_right` | Right wrist camera | `418643070357` | `Intel_R__RealSense_TM__Depth_Camera_405_..._418643070357` |
| `/dev/cam_wrist_left` | Left wrist camera | `418643070428` | `Intel_R__RealSense_TM__Depth_Camera_405_..._418643070428` |

---

## How It Works

A udev rule is a single line of comma-separated match and assignment clauses:

| Clause | Type | Meaning |
|---|---|---|
| `SUBSYSTEM=="video4linux"` | Match | Only apply to video devices |
| `ENV{ID_SERIAL}=="..."` | Match | Match this specific physical camera by serial |
| `ATTR{index}=="4"` | Match | Match only the RGB stream node (index 4 of 6) |
| `SYMLINK+="cam_high"` | Assign | Create `/dev/cam_high` pointing to the matched node |

The `+=` operator appends the symlink without removing the original `/dev/videoX` entry.

---

## Troubleshooting

**Symlinks not created after reload:**
```bash
# Test the rule against a specific device without rebooting
sudo udevadm test $(udevadm info --query=path --name=/dev/video4)
```

**Wrong stream (not RGB):**  
Re-run `ffplay /dev/videoX` to find the correct RGB node, then update `ATTR{index}` accordingly.

**On a new machine — serials changed:**  
Re-run Step 1 to get the new serials, update the rules file, and reload.
