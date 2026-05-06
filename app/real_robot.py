"""Real-hardware adapter for the bimanual openarm follower.

Same surface as `MujocoBiOpenarm` (connect / get_state / send_action / step /
disconnect) so `run_remote_pi05.py` can swap between sim and real with one flag.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mujoco_robot import ACTION_FEATURE_NAMES

log = logging.getLogger("real_robot")


def auto_find_arm_ports() -> tuple[str | None, str | None]:
    """Return (left_port, right_port) by env vars, falling back to /dev scan.

    macOS exposes USB serial ports as `/dev/cu.usbserial-*`; Linux as
    `/dev/ttyUSB*` or `/dev/ttyACM*`. We can't reliably distinguish left vs
    right from the device name alone, so the user must either set the env
    vars LEFT_FOLLOWER / RIGHT_FOLLOWER or accept the alphabetical order.
    """
    left = os.environ.get("LEFT_FOLLOWER")
    right = os.environ.get("RIGHT_FOLLOWER")
    if left and right:
        return left, right

    candidates = sorted(
        glob.glob("/dev/cu.usbserial-*")
        + glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
    )
    if len(candidates) >= 2:
        log.warning(
            "auto-picked arm ports by alphabetical order: left=%s right=%s "
            "(set LEFT_FOLLOWER/RIGHT_FOLLOWER to override)",
            candidates[0], candidates[1],
        )
        return candidates[0], candidates[1]
    return left, right


@dataclass
class RealBiOpenarmConfig:
    left_port: str
    right_port: str
    calibration_dir: Path
    can_interface: str = field(default_factory=lambda: os.environ.get("CAN_INTERFACE", "auto"))
    max_relative_target: float = 5.0
    robot_id: str = "my_bimanual_follower"


class RealBiOpenarm:
    """Wraps lerobot's BiOpenArmFollower with the same API as MujocoBiOpenarm."""

    # Attributes referenced by run_remote_pi05.py for the viewer; None disables it.
    model = None
    data = None

    def __init__(self, cfg: RealBiOpenarmConfig):
        if cfg.can_interface == "pcan":
            _patch_pcan_can_fd_kwargs()

        from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenArmFollower
        from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import (
            BiOpenArmFollowerConfig,
        )
        from lerobot.robots.openarm_follower.config_openarm_follower import (
            OpenArmFollowerConfigBase,
        )

        self.cfg = cfg
        self.robot = BiOpenArmFollower(BiOpenArmFollowerConfig(
            id=cfg.robot_id,
            calibration_dir=Path(cfg.calibration_dir),
            left_arm_config=OpenArmFollowerConfigBase(
                port=cfg.left_port, side="left",
                can_interface=cfg.can_interface,
                max_relative_target=cfg.max_relative_target,
            ),
            right_arm_config=OpenArmFollowerConfigBase(
                port=cfg.right_port, side="right",
                can_interface=cfg.can_interface,
                max_relative_target=cfg.max_relative_target,
            ),
        ))

    def connect(self):
        log.info("connecting bi_openarm_follower (left=%s right=%s)",
                 self.cfg.left_port, self.cfg.right_port)
        self.robot.connect()

    def disconnect(self):
        try:
            self.robot.disconnect()
        except Exception:
            log.exception("robot.disconnect failed")

    def get_state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[k]) for k in ACTION_FEATURE_NAMES], dtype=np.float32)

    def send_action(self, action_deg: np.ndarray):
        self.robot.send_action({k: float(v) for k, v in zip(ACTION_FEATURE_NAMES, action_deg)})

    def step(self, dt: float):
        # Real hardware advances on its own; nothing to do.
        return


def _patch_pcan_can_fd_kwargs() -> None:
    """python-can's PCAN backend on macOS/libPCBUSB rejects bitrate/data_bitrate.

    The backend accepts explicit timing parameters instead. LeRobot's Damiao bus
    passes the SocketCAN-style shorthand, so translate it before Bus creation.
    """
    import can

    if getattr(can.interface.Bus, "_openarm_pcan_patch", False):
        return

    original_bus = can.interface.Bus

    def bus_with_pcan_timing(*args, **kwargs):
        if kwargs.get("interface") == "pcan" and kwargs.get("fd"):
            kwargs.pop("bitrate", None)
            kwargs.pop("data_bitrate", None)
            kwargs.setdefault("f_clock_mhz", 80)
            kwargs.setdefault("nom_brp", 1)
            kwargs.setdefault("nom_tseg1", 63)
            kwargs.setdefault("nom_tseg2", 16)
            kwargs.setdefault("nom_sjw", 16)
            kwargs.setdefault("data_brp", 1)
            kwargs.setdefault("data_tseg1", 11)
            kwargs.setdefault("data_tseg2", 4)
            kwargs.setdefault("data_sjw", 4)
        return original_bus(*args, **kwargs)

    bus_with_pcan_timing._openarm_pcan_patch = True
    can.interface.Bus = bus_with_pcan_timing
