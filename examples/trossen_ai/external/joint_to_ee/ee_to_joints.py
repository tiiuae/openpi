"""EE -> joints via placo IK, plus joints -> EE obs FK. Needs URDF + placo.

decode_chunk converts an (N, 16) absolute-EE-quat action chunk (robot-base frame)
into an (N, 14) joint chunk (radians + gripper meters) by per-arm IK, seeded for
continuity. On an unreachable / failed solve it holds the last valid joints.
"""
import numpy as np

from . import constants as C
from .ee_frames import pose8_to_arm_se3
from .kinematics import fk_pose8


class EEToJointsConverter:
    def __init__(self, kin, position_weight: float = 1.0, orientation_weight: float = 0.01,
                 pos_tol_m: float = 1e-3, ik_max_iters: int = 50):
        self.kin = kin
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.pos_tol_m = pos_tol_m
        self.ik_max_iters = ik_max_iters

    # ---- observation side: joints -> EE state (reuses existing FK) ----
    def joints14_to_ee16(self, joints14: np.ndarray) -> np.ndarray:
        q = np.asarray(joints14, dtype=np.float64).flatten()
        left = fk_pose8(self.kin, q, C.OBS_LEFT_JOINTS, C.LEFT_MOUNT_XYZ, q[C.LEFT_GRIPPER_IDX])
        right = fk_pose8(self.kin, q, C.OBS_RIGHT_JOINTS, C.RIGHT_MOUNT_XYZ, q[C.RIGHT_GRIPPER_IDX])
        return np.concatenate([left, right]).astype(np.float32)

    # ---- action side: EE -> joints ----
    def _ik_arm(self, pose8, mount_xyz, seed_deg6, fallback_rad6):
        """Return (arm_joints_rad6, grip_m, next_seed_deg6, ok).

        lerobot's `inverse_kinematics` performs a single velocity step, so it is
        iterated here (feeding its output back as the seed) until the FK of the
        solution is within `pos_tol_m` of the target or `ik_max_iters` is reached.
        """
        T, grip_m = pose8_to_arm_se3(pose8, mount_xyz)
        joints_deg = np.asarray(seed_deg6, dtype=np.float64).flatten()[:6]
        pos_err = np.inf
        for _ in range(self.ik_max_iters):
            joints_deg = self.kin.inverse_kinematics(
                joints_deg,
                T,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight,
            )
            joints_deg = np.asarray(joints_deg, dtype=np.float64).flatten()[:6]
            if not np.all(np.isfinite(joints_deg)):
                break
            T_check = self.kin.forward_kinematics(joints_deg)
            pos_err = float(np.linalg.norm(T_check[:3, 3] - T[:3, 3]))
            if pos_err <= self.pos_tol_m:
                break
        # hold last valid joints on non-convergence / unreachable target
        if not np.all(np.isfinite(joints_deg)) or pos_err > self.pos_tol_m:
            return (np.asarray(fallback_rad6, dtype=np.float32), grip_m,
                    np.rad2deg(fallback_rad6), False)
        joints_rad = np.deg2rad(joints_deg).astype(np.float32)
        return joints_rad, grip_m, joints_deg, True

    def decode_chunk(self, raw_chunk: np.ndarray, current_joints14: np.ndarray) -> np.ndarray:
        chunk16 = np.asarray(raw_chunk, dtype=np.float64)[:, :16]
        cur = np.asarray(current_joints14, dtype=np.float64).flatten()
        out = np.zeros((len(chunk16), 14), dtype=np.float32)

        seed_l = np.rad2deg(cur[C.OBS_LEFT_JOINTS])
        seed_r = np.rad2deg(cur[C.OBS_RIGHT_JOINTS])
        fb_l = cur[C.OBS_LEFT_JOINTS].astype(np.float32)
        fb_r = cur[C.OBS_RIGHT_JOINTS].astype(np.float32)

        for i, row in enumerate(chunk16):
            jl, gl, seed_l, _ = self._ik_arm(row[:8], C.LEFT_MOUNT_XYZ, seed_l, fb_l)
            jr, gr, seed_r, _ = self._ik_arm(row[8:16], C.RIGHT_MOUNT_XYZ, seed_r, fb_r)
            full = np.zeros(14, dtype=np.float32)
            full[C.OBS_LEFT_JOINTS] = jl
            full[C.LEFT_GRIPPER_IDX] = gl
            full[C.OBS_RIGHT_JOINTS] = jr
            full[C.RIGHT_GRIPPER_IDX] = gr
            out[i] = full
            fb_l, fb_r = jl, jr  # next-row fallback = this row's joints

        # Kill residual ±2π/branch wraps on the revolute joints: pick, per joint,
        # the 2π-equivalent angle nearest the previous frame. This removes the
        # single-frame "line" jumps (a wrist flipping by ~2π) that look like an
        # aggressive move while the EE pose is actually unchanged. Grippers
        # (prismatic, metres) are left alone.
        rev_idx = list(C.OBS_LEFT_JOINTS) + list(C.OBS_RIGHT_JOINTS)
        if len(out) > 1:
            out[:, rev_idx] = np.unwrap(out[:, rev_idx].astype(np.float64),
                                        axis=0).astype(np.float32)
        return out
