import dataclasses
import enum
import logging
import socket
from typing import Literal

import tyro

from openpi.policies import cogact_policy as _cogact_policy
from openpi.policies import openvla_policy as _openvla_policy
from openpi.policies import openvlaoft_policy as _openvlaoft_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    FALCONVLA_ALOHA = "falconvla_aloha"
    ACT_ALOHA = "act_aloha"
    OPENVLA = "openvla"
    OPENVLA_OFT = "openvla-oft"
    COGACT = "cogact"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class AutoCheckpoint:
    """Load a FalconVLA policy from a checkpoint directory, auto-detecting its config.

    `action_dim`, `action_horizon`, `unnorm_key`, and proprio settings are read directly from the
    checkpoint's own config.json / norm_stats.json / tokenizer files -- no matching `_CONFIGS`
    entry needed. FalconVLA checkpoints only.
    """

    # FalconVLA checkpoint directory (e.g. a directory under VLA_MODELS/).
    dir: str
    # Overrides -- only needed when auto-detection is ambiguous (e.g. multiple norm_stats.json
    # keys) or to force a non-default value.
    unnorm_key: str | None = None
    use_proprio: bool | None = None
    proprio_mode: Literal["tokens", "film"] | None = None


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8800
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | AutoCheckpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.FALCONVLA_ALOHA: Checkpoint(
        config="FalconVLA-AD14-H25",
        dir="/models/FalconVLA-8B-aidrc_cups_manipulation-er-3v-p",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.OPENVLA: Checkpoint(
        config="openvla",
        dir="",
    ),
    EnvMode.OPENVLA_OFT: Checkpoint(
        config="openvla-oft",
        dir="",
    ),
    EnvMode.COGACT: Checkpoint(
        config="cogact",
        dir="",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    # Route REST-backed environments to their client policies. FalconVLA (in-process) needs no
    # special case here — it dispatches through create_trained_policy on its model type.
    if args.env == EnvMode.OPENVLA:
        logging.info("Using OpenVLAClientPolicy (REST backend)")
        return _openvla_policy.OpenVLAClientPolicy(
            server_url="http://localhost:8000/act",
            timeout=10.0,
            unnorm_key="aidrc_cups_manipulation_14",
            default_prompt=args.default_prompt,
        )

    # 3. Route to OpenVLA-OFT
    if args.env == EnvMode.OPENVLA_OFT:
        logging.info("Using OpenVLA-OFTClientPolicy (REST backend)")
        return _openvlaoft_policy.OpenVLAOFTClientPolicy(
            server_url="http://localhost:8777/act",
            timeout=10.0,
            unnorm_key="aidrc_cups_manipulation_14",
            default_prompt=args.default_prompt,
        )

    # 4. Route to CogACT
    if args.env == EnvMode.COGACT:
        logging.info("Using CogACTClientPolicy (REST backend)")
        return _cogact_policy.CogACTClientPolicy(
            server_url="http://localhost:8777/api/inference",
            timeout=10.0,
            unnorm_key="aidrc_cups_manipulation_14",
            default_prompt=args.default_prompt,
        )
    #  Check if a specific policy checkpoint was provided
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case AutoCheckpoint():
            overrides = {
                k: v
                for k, v in {
                    "unnorm_key": args.policy.unnorm_key,
                    "use_proprio": args.policy.use_proprio,
                    "proprio_mode": args.policy.proprio_mode,
                }.items()
                if v is not None
            }
            return _policy_config.create_trained_policy(
                _config.get_falconvla_train_config(args.policy.dir, **overrides),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
