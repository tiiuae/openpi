"""Action-space adapters: pluggable observation-state builder + action decoder.

The bridge is action-space agnostic. A JointAdapter keeps today's behavior. An
EEAdapter (Task 4) FKs the observation into EE space and IK-decodes EE action
chunks back into joints, so everything downstream of the adapter is unchanged.
"""

from __future__ import annotations

import numpy as np

JOINT_DIM = 14


def extract_joints(obs_dict: dict) -> np.ndarray:
    """Ordered 14-D joint vector from a robot observation dict (`*.pos` keys)."""
    keys = [k for k in obs_dict if k.endswith(".pos")]
    return np.array([obs_dict[k] for k in keys], dtype=np.float64)


class ActionSpaceAdapter:
    """Interface between the policy server and the joint-space control loop."""

    #: dimensionality of the `state` vector sent to the policy
    state_dim: int

    def build_state(self, obs_dict: dict) -> np.ndarray:
        raise NotImplementedError

    def decode_chunk(self, raw_chunk: np.ndarray, current_joints14: np.ndarray) -> np.ndarray:
        """Map a model action chunk to an (N, 14) joint chunk (rad + gripper m)."""
        raise NotImplementedError


class JointAdapter(ActionSpaceAdapter):
    """Identity adapter: the policy already speaks joint space."""

    state_dim = JOINT_DIM

    def build_state(self, obs_dict: dict) -> np.ndarray:
        return extract_joints(obs_dict)

    def decode_chunk(self, raw_chunk: np.ndarray, current_joints14: np.ndarray) -> np.ndarray:
        return np.asarray(raw_chunk)[:, :JOINT_DIM]


class EEAdapter(ActionSpaceAdapter):
    """EE-space policy: FK observation into EE, IK-decode EE actions into joints."""

    state_dim = 16

    def __init__(self, converter):
        # converter: external.joint_to_ee.ee_to_joints.EEToJointsConverter
        self._converter = converter

    def build_state(self, obs_dict: dict) -> np.ndarray:
        joints14 = extract_joints(obs_dict)
        return self._converter.joints14_to_ee16(joints14)

    def decode_chunk(self, raw_chunk: np.ndarray, current_joints14: np.ndarray) -> np.ndarray:
        return self._converter.decode_chunk(raw_chunk, current_joints14)
