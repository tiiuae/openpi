#!/usr/bin/env python3
"""Trossen Arm <-> OpenPI bridge — end-effector (EE) entrypoint.

The policy outputs absolute 8-D EE poses per arm ([x,y,z,qw,qx,qy,qz,grip], robot
-base frame). This entrypoint FKs observations into EE space and IK-decodes EE
actions back into joints (see end_effector_support.md, Option 1). Synchronous
inference only (async EE decoding is not yet supported).
"""
import argparse

from adapters import EEAdapter
from external.joint_to_ee.ee_to_joints import EEToJointsConverter
from external.joint_to_ee.kinematics import make_kinematics
from main import build_parser as build_joint_parser
from trossen_bridge import TrossenOpenPIBridge


def build_parser() -> argparse.ArgumentParser:
    parser = build_joint_parser()
    parser.description = "Trossen AI <-> OpenPI Policy Server Bridge (end-effector space)"
    parser.add_argument("--ik_orientation_weight", type=float, default=0.01,
                        help="placo IK orientation weight (raise for tighter rotation tracking)")
    parser.add_argument("--ik_pos_tol_m", type=float, default=1e-3,
                        help="IK convergence + failure tolerance (m): loop refines until FK error <= this; above it after ik_max_iters -> hold last")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.async_inference:
        raise SystemExit("--async_inference is not supported in EE mode yet (see plan scope note).")

    converter = EEToJointsConverter(
        make_kinematics(),
        orientation_weight=args.ik_orientation_weight,
        pos_tol_m=args.ik_pos_tol_m,
    )
    adapter = EEAdapter(converter)

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        rate_of_inference=args.rate_of_inference,
        ensemble_type=args.ensemble_type,
        cogact_mode=args.cogact_mode,
        async_inference=False,
        log_dir=args.log_dir,
        use_left_arm_only=args.use_left_arm_only,
        use_right_arm_only=args.use_right_arm_only,
        starvla=args.starvla,
        adapter=adapter,
    )
    bridge.autonomous_mode(task_prompt=args.task_prompt)
    bridge.cleanup()


if __name__ == "__main__":
    main()
