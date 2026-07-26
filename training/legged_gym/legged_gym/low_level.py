"""Validated low-level policy contracts shared by training and navigation."""

from __future__ import annotations

import hashlib
import json
import math
import os

import torch


BLIND_STAIR_OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",
    "joint_pos_error",
    "joint_vel",
    "last_action",
)
BLIND_STAIR_OBSERVATION_LAYOUT = (
    {"name": "base_ang_vel", "start": 0, "end": 3},
    {"name": "projected_gravity", "start": 3, "end": 6},
    {"name": "velocity_command", "start": 6, "end": 9},
    {"name": "joint_pos_error", "start": 9, "end": 21},
    {"name": "joint_vel", "start": 21, "end": 33},
    {"name": "last_action", "start": 33, "end": 45},
)
GO2_EXTERNAL_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
GO2_INTERNAL_TO_EXTERNAL_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


def build_blind_stair_observation(
    base_ang_vel,
    projected_gravity,
    velocity_command,
    joint_pos_error,
    joint_vel,
    last_action,
    *,
    lin_vel_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
):
    """Build the single 45-D observation contract used for training and execution."""
    terms = (
        base_ang_vel * ang_vel_scale,
        projected_gravity,
        velocity_command
        * velocity_command.new_tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale]),
        joint_pos_error * dof_pos_scale,
        joint_vel * dof_vel_scale,
        last_action,
    )
    expected_widths = (3, 3, 3, 12, 12, 12)
    batch_size = base_ang_vel.shape[0]
    for name, value, width in zip(BLIND_STAIR_OBSERVATION_TERMS, terms, expected_widths):
        if value.ndim != 2 or value.shape != (batch_size, width):
            raise ValueError(
                f"Blind stair observation term {name} must have shape "
                f"({batch_size}, {width}), got {tuple(value.shape)}"
            )
    return torch.cat(terms, dim=-1)


def _joint_values(value, name):
    if value is None:
        raise ValueError(f"Blind stair {name} is missing")
    if isinstance(value, (int, float)):
        return [float(value)] * 12
    values = [float(item) for item in value]
    if len(values) != 12:
        raise ValueError(f"Blind stair {name} must contain 12 joint values")
    return values


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_blind_stair_metadata(
    model_path,
    checkpoint_path,
    default_joint_angles,
    control_frequency_hz=50.0,
    action_scale=0.25,
    stiffness=30.0,
    damping=0.75,
):
    return {
        "schema_version": 1,
        "input_dim": 45,
        "output_dim": 12,
        "observation_terms": list(BLIND_STAIR_OBSERVATION_TERMS),
        "observation_layout": list(BLIND_STAIR_OBSERVATION_LAYOUT),
        "joint_order": list(GO2_EXTERNAL_JOINT_ORDER),
        "observation_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "default_joint_angles": [float(value) for value in default_joint_angles],
        "control": {
            "frequency_hz": float(control_frequency_hz),
            "action_scale": float(action_scale),
            "stiffness": _joint_values(stiffness, "stiffness"),
            "damping": _joint_values(damping, "damping"),
        },
        "model_sha256": sha256_file(model_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


class BlindStairPolicy:
    """TorchScript policy that refuses mismatched SEA control contracts."""

    def __init__(self, model_path, metadata_path, device, expected):
        self.model_path = os.path.abspath(model_path)
        self.metadata_path = os.path.abspath(metadata_path or model_path + ".json")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Blind stair locomotion model not found: {self.model_path}")
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(f"Blind stair locomotion metadata not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        self._validate_metadata(expected)
        self.model = torch.jit.load(self.model_path, map_location=device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.zeros(2, 45, device=device))
        if output.shape != (2, 12):
            raise ValueError(
                f"Blind stair locomotion model must map [batch, 45] to [batch, 12], got {tuple(output.shape)}"
            )

    @staticmethod
    def _close(actual, expected, name):
        if actual is None:
            raise ValueError(f"Blind stair metadata is missing {name}")
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-6
        ):
            raise ValueError(f"Blind stair metadata {name}={actual} does not match runtime {expected}")

    @classmethod
    def _close_joint_values(cls, actual, expected, name):
        actual_values = _joint_values(actual, f"metadata {name}")
        expected_values = _joint_values(expected, f"runtime {name}")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_values, expected_values)
        ):
            cls._close(actual_value, expected_value, f"{name}[{index}]")

    def _validate_metadata(self, expected):
        metadata = self.metadata
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported blind stair metadata schema")
        if metadata.get("input_dim") != 45 or metadata.get("output_dim") != 12:
            raise ValueError("Blind stair metadata must declare a 45 -> 12 policy")
        if tuple(metadata.get("observation_terms", ())) != BLIND_STAIR_OBSERVATION_TERMS:
            raise ValueError("Blind stair observation order does not match SEA")
        if tuple(metadata.get("observation_layout", ())) != BLIND_STAIR_OBSERVATION_LAYOUT:
            raise ValueError("Blind stair observation layout does not match SEA")
        if tuple(metadata.get("joint_order", ())) != GO2_EXTERNAL_JOINT_ORDER:
            raise ValueError("Blind stair joint order does not match SEA")
        if tuple(expected.get("joint_order", ())) != GO2_EXTERNAL_JOINT_ORDER:
            raise ValueError("Runtime Go2 URDF joint order does not match blind stair policy")
        if metadata.get("model_sha256") != sha256_file(self.model_path):
            raise ValueError("Blind stair model SHA-256 does not match metadata")
        scales = metadata.get("observation_scales", {})
        for name, value in expected["observation_scales"].items():
            self._close(scales.get(name), value, f"observation_scales.{name}")
        control = metadata.get("control", {})
        for name in ("frequency_hz", "action_scale"):
            self._close(control.get(name), expected["control"][name], f"control.{name}")
        for name in ("stiffness", "damping"):
            self._close_joint_values(
                control.get(name), expected["control"][name], f"control.{name}"
            )
        actual_angles = metadata.get("default_joint_angles", ())
        expected_angles = expected["default_joint_angles"]
        if len(actual_angles) != len(expected_angles):
            raise ValueError("Blind stair default joint angle count does not match SEA")
        for index, (actual, value) in enumerate(zip(actual_angles, expected_angles)):
            self._close(actual, value, f"default_joint_angles[{index}]")

    def __call__(self, observations):
        if observations.ndim != 2 or observations.shape[1] != 45:
            raise ValueError(f"Blind stair policy expects [batch, 45], got {tuple(observations.shape)}")
        with torch.no_grad():
            return self.model(observations)
