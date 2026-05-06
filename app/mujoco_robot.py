"""MuJoCo-backed stand-in for `BiOpenArmFollower` used to drive a pi05 policy.

Joint conventions (chosen to match the dataset/calibration the policy was trained on):

* Arm joints 1..7 (per side):
    sim native = radians;
    policy native = **degrees** (calibration json range_min/max are ±90 in deg).
* Gripper (per side):
    sim native = slide joint in metres, ctrlrange [0, 0.044];
    policy native = `gripper.pos` in degrees with limit (-65.0, 0.0).
    Mapping is linear: slide=0.044 (open) <-> 0°,  slide=0 (closed) <-> -65°.

If the policy seems to drive the arm in the wrong direction or saturates, the only
file to edit is this one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Order in which the model emits and consumes action features (from config.json).
ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
    "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
    "left_joint_1.pos",  "left_joint_2.pos",  "left_joint_3.pos",  "left_joint_4.pos",
    "left_joint_5.pos",  "left_joint_6.pos",  "left_joint_7.pos",  "left_gripper.pos",
)

GRIPPER_OPEN_M = 0.044   # slide pos when fully open (matches XML ctrlrange upper)
GRIPPER_CLOSED_M = 0.0
GRIPPER_OPEN_DEG = 0.0
GRIPPER_CLOSED_DEG = -65.0


def _gripper_m_to_deg(s: float) -> float:
    t = (s - GRIPPER_CLOSED_M) / (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
    return GRIPPER_CLOSED_DEG + t * (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG)


def _gripper_deg_to_m(d: float) -> float:
    t = (d - GRIPPER_CLOSED_DEG) / (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG)
    return float(np.clip(GRIPPER_CLOSED_M + t * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M),
                         GRIPPER_CLOSED_M, GRIPPER_OPEN_M))


@dataclass
class MujocoBiOpenarmConfig:
    mjcf_path: Path
    initial_qpos: dict[str, float] = field(default_factory=dict)  # joint_name -> radians
    sim_substeps: int = 1  # extra mj_step per send_action call beyond the control tick


class MujocoBiOpenarm:
    def __init__(self, cfg: MujocoBiOpenarmConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self._joint_qpos_addr: dict[str, int] = {}
        for name in self._all_joint_names():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint {name!r} not found in {cfg.mjcf_path}")
            self._joint_qpos_addr[name] = self.model.jnt_qposadr[jid]

        self._actuator_id: dict[str, int] = {}
        for name in self._actuator_names():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator {name!r} not found")
            self._actuator_id[name] = aid

        self.is_connected = False

    # ---- joint / actuator naming ---------------------------------------------------

    @staticmethod
    def _all_joint_names() -> list[str]:
        out = []
        for side in ("left", "right"):
            for j in range(1, 8):
                out.append(f"openarm_{side}_joint{j}")
            for f in (1, 2):
                out.append(f"openarm_{side}_finger_joint{f}")
        return out

    @staticmethod
    def _actuator_names() -> list[str]:
        # Arm motors only — finger control is set through qpos directly because
        # the XML mixes <motor> (left) with <position> (right) actuators for fingers.
        out = []
        for side in ("left", "right"):
            for j in range(1, 8):
                out.append(f"{side}_joint{j}_ctrl")
        return out

    # ---- lifecycle -----------------------------------------------------------------

    def connect(self):
        for name, q in self.cfg.initial_qpos.items():
            self.data.qpos[self._joint_qpos_addr[name]] = q
        mujoco.mj_forward(self.model, self.data)
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    # ---- observation ---------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """16-d state in policy space (degrees for joints, mapped degrees for grippers)."""
        out = np.zeros(16, dtype=np.float32)
        rad2deg = 180.0 / np.pi
        for i, name in enumerate(ACTION_FEATURE_NAMES):
            side, what = name.split("_", 1)  # "right" / "right_joint_3.pos"
            base = name.removesuffix(".pos")
            if "joint" in base:
                jnum = int(base.split("_joint_")[1])
                q = self.data.qpos[self._joint_qpos_addr[f"openarm_{side}_joint{jnum}"]]
                out[i] = float(q) * rad2deg
            else:  # gripper
                # average of finger_joint1 and finger_joint2 slide positions
                f1 = self.data.qpos[self._joint_qpos_addr[f"openarm_{side}_finger_joint1"]]
                f2 = self.data.qpos[self._joint_qpos_addr[f"openarm_{side}_finger_joint2"]]
                out[i] = _gripper_m_to_deg(0.5 * (float(f1) + float(f2)))
        return out

    # ---- action --------------------------------------------------------------------

    def send_action(self, action_deg: np.ndarray):
        """Write a single timestep target (16-d, policy units)."""
        deg2rad = np.pi / 180.0
        for i, name in enumerate(ACTION_FEATURE_NAMES):
            side = "right" if name.startswith("right_") else "left"
            base = name.removesuffix(".pos")
            if "joint" in base:
                jnum = int(base.split("_joint_")[1])
                ctrl_id = self._actuator_id[f"{side}_joint{jnum}_ctrl"]
                self.data.ctrl[ctrl_id] = float(action_deg[i]) * deg2rad
            else:  # gripper — write qpos directly (see class docstring)
                slide = _gripper_deg_to_m(float(action_deg[i]))
                self.data.qpos[self._joint_qpos_addr[f"openarm_{side}_finger_joint1"]] = slide
                self.data.qpos[self._joint_qpos_addr[f"openarm_{side}_finger_joint2"]] = slide

    def step(self, dt: float):
        """Advance the simulation by approximately `dt` seconds of wall time."""
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
