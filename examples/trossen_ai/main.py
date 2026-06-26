#!/usr/bin/env python3
"""Trossen Arm <-> OpenPI bridge — joint-space entrypoint.

See main_ee.py for the end-effector variant. Shared control loop lives in
trossen_bridge.py.
"""

import argparse

from adapters import JointAdapter
from trossen_bridge import TrossenOpenPIBridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trossen AI <-> OpenPI Policy Server Bridge (joint space)")
    parser.add_argument("--policy_host", default="192.168.50.174", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8800, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=25, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="autonomous (execute) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--action_chunk_size", type=int, default=25, help="Actions predicted per inference")
    parser.add_argument("--rate_of_inference", type=int, default=20, help="Control steps between inferences")
    parser.add_argument(
        "--ensemble_type", choices=["exp", "cogact", "none"], default="exp", help="Action ensemble strategy"
    )
    parser.add_argument(
        "--cogact_mode",
        choices=["cogact", "latest", "hybrid"],
        default="cogact",
        help="CogACT weighting mode (only used with --ensemble_type cogact)",
    )
    parser.add_argument("--log_dir", default=None, help="Directory for per-episode overlap JSON")
    parser.add_argument("--async_inference", action="store_true", help="Background-thread inference (needs ensemble)")
    parser.add_argument("--starvla", action="store_true", help="StarVLA 224x224 PIL resizing")
    parser.add_argument("--use_left_arm_only", action="store_true", help="Only move the left arm")
    parser.add_argument("--use_right_arm_only", action="store_true", help="Only move the right arm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        async_inference=args.async_inference,
        log_dir=args.log_dir,
        use_left_arm_only=args.use_left_arm_only,
        use_right_arm_only=args.use_right_arm_only,
        starvla=args.starvla,
        adapter=JointAdapter(),
    )
    bridge.autonomous_mode(task_prompt=args.task_prompt)
    bridge.cleanup()


if __name__ == "__main__":
    main()
