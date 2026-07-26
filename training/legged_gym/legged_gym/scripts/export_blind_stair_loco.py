"""Export a blind stair locomotion checkpoint with its runtime contract."""

import argparse
import json
import os

import torch

from legged_gym.low_level import (
    BlindStairPolicy,
    GO2_EXTERNAL_JOINT_ORDER,
    build_blind_stair_metadata,
)
from rsl_rl.modules import BlindLocomotionActorCritic


DEFAULT_JOINT_ANGLES = (-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy = BlindLocomotionActorCritic(
        num_actions=12,
        num_actor_obs=45,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    actor = policy.actor.eval().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.jit.script(actor).save(args.output)

    metadata_path = args.metadata or args.output + ".json"
    metadata = build_blind_stair_metadata(
        args.output,
        args.checkpoint,
        DEFAULT_JOINT_ANGLES,
    )
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    BlindStairPolicy(
        args.output,
        metadata_path,
        "cpu",
        {
            "observation_scales": metadata["observation_scales"],
            "joint_order": GO2_EXTERNAL_JOINT_ORDER,
            "default_joint_angles": DEFAULT_JOINT_ANGLES,
            "control": metadata["control"],
        },
    )
    print(f"Exported policy: {os.path.abspath(args.output)}")
    print(f"Exported metadata: {os.path.abspath(metadata_path)}")


if __name__ == "__main__":
    main()
