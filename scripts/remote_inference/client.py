"""Local client: reads bimanual openarm + 3 cameras, ships obs to a remote pi05
server, executes the returned action chunk at FPS, prefetches the next chunk.

Mirrors the device wiring from `scripts/run_pi05_openarm_mps.sh`.

Example:
    LEFT_OPENARM_PORT=/dev/cu.usbserial-left \
    RIGHT_OPENARM_PORT=/dev/cu.usbserial-right \
    SERVER_URL=ws://127.0.0.1:8765 \
    python client.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import numpy as np
import websockets

from protocol import (
    ACTION_DIM,
    IMAGE_KEYS,
    STATE_DIM,
    decode_response,
    encode_request,
    jpeg_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("client")


def build_robot(args):
    """Instantiate the bimanual openarm follower exactly like lerobot-record does."""
    from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenArmFollower
    from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import (
        BiOpenArmFollowerConfig,
    )
    from lerobot.robots.openarm_follower.config_openarm_follower import (
        OpenArmFollowerConfig,
    )
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cameras = {
        "left_wrist": OpenCVCameraConfig(
            index_or_path=int(os.environ.get("LEFT_WRIST_CAMERA", "0")),
            width=1280, height=720, fps=args.fps,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=int(os.environ.get("RIGHT_WRIST_CAMERA", "1")),
            width=1280, height=720, fps=args.fps,
        ),
        "base": OpenCVCameraConfig(
            index_or_path=int(os.environ.get("BASE_CAMERA", "2")),
            width=640, height=480, fps=args.fps,
        ),
    }
    can_iface = os.environ["CAN_INTERFACE"] if "CAN_INTERFACE" in os.environ else "auto"
    cfg = BiOpenArmFollowerConfig(
        id=args.robot_id,
        calibration_dir=Path(args.calibration_dir),
        left_arm_config=OpenArmFollowerConfig(
            port=os.environ["LEFT_OPENARM_PORT"],
            side="left",
            can_interface=can_iface,
            max_relative_target=5,
        ),
        right_arm_config=OpenArmFollowerConfig(
            port=os.environ["RIGHT_OPENARM_PORT"],
            side="right",
            can_interface=can_iface,
            max_relative_target=5,
        ),
        cameras=cameras,
    )
    return BiOpenArmFollower(cfg)


# Joint order the policy was trained with (from config.json action_feature_names).
# Robot observation/action dicts use "<joint>.pos" keys; arrange them in this order
# to build the 16-d state vector and to translate actions back into a dict.
ACTION_FEATURE_NAMES = [
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
    "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos", "left_joint_4.pos",
    "left_joint_5.pos", "left_joint_6.pos", "left_joint_7.pos", "left_gripper.pos",
]


def obs_to_state_and_images(obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state = np.array([float(obs[k]) for k in ACTION_FEATURE_NAMES], dtype=np.float32)
    images = {k: obs[k] for k in IMAGE_KEYS}  # already HxWx3 uint8 RGB from lerobot cameras
    return state, images


def action_vec_to_dict(action: np.ndarray) -> dict[str, float]:
    return {name: float(v) for name, v in zip(ACTION_FEATURE_NAMES, action)}


async def control_loop(robot, ws, task: str, fps: int, chunk_reuse: int):
    """Run policy execution. `chunk_reuse` is how many actions of a chunk to play
    before requesting the next one (must be < chunk_size to allow prefetching)."""
    period = 1.0 / fps

    async def request_chunk(state, jpeg_imgs):
        await ws.send(encode_request(state, jpeg_imgs, task))
        reply = await ws.recv()
        actions, infer_ms = decode_response(reply)
        return actions, infer_ms

    # --- Prime: get the first chunk synchronously ---
    obs = robot.get_observation()
    state, imgs = obs_to_state_and_images(obs)
    jpeg_imgs = {k: jpeg_encode(imgs[k]) for k in IMAGE_KEYS}
    t0 = time.perf_counter()
    chunk, infer_ms = await request_chunk(state, jpeg_imgs)
    log.info("first chunk: shape=%s infer=%.1fms total=%.1fms",
             chunk.shape, infer_ms, (time.perf_counter() - t0) * 1000)

    chunk_size = chunk.shape[0]
    chunk_reuse = min(chunk_reuse, chunk_size - 1)
    log.info("control loop: fps=%d chunk_size=%d execute_per_chunk=%d", fps, chunk_size, chunk_reuse)

    pending: asyncio.Task | None = None
    next_chunk: np.ndarray | None = None
    step_in_chunk = 0
    next_tick = time.perf_counter()

    try:
        while True:
            # Send action for this tick
            action_vec = chunk[step_in_chunk]
            robot.send_action(action_vec_to_dict(action_vec))
            step_in_chunk += 1

            # Halfway through the chunk, kick off the next inference call.
            if pending is None and step_in_chunk >= chunk_reuse // 2:
                obs = robot.get_observation()
                s, im = obs_to_state_and_images(obs)
                jp = {k: jpeg_encode(im[k]) for k in IMAGE_KEYS}
                pending = asyncio.create_task(request_chunk(s, jp))

            # End of usable portion: swap in the prefetched chunk.
            if step_in_chunk >= chunk_reuse:
                if pending is None:
                    obs = robot.get_observation()
                    s, im = obs_to_state_and_images(obs)
                    jp = {k: jpeg_encode(im[k]) for k in IMAGE_KEYS}
                    pending = asyncio.create_task(request_chunk(s, jp))
                t_wait = time.perf_counter()
                next_chunk, infer_ms = await pending
                log.info("swap chunk: infer=%.1fms wait_blocked=%.1fms",
                         infer_ms, (time.perf_counter() - t_wait) * 1000)
                pending = None
                chunk = next_chunk
                step_in_chunk = 0

            # Pace the loop
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                # Fell behind — log and reset cadence.
                if sleep < -period:
                    log.warning("loop fell behind by %.1f ms", -sleep * 1000)
                next_tick = time.perf_counter()
    finally:
        if pending is not None:
            pending.cancel()


async def main_async(args):
    robot = build_robot(args)
    log.info("Connecting robot...")
    robot.connect()
    try:
        log.info("Connecting to %s ...", args.server_url)
        async with websockets.connect(
            args.server_url, max_size=32 * 1024 * 1024, ping_interval=20, ping_timeout=20,
        ) as ws:
            await control_loop(robot, ws, args.task, args.fps, args.chunk_reuse)
    finally:
        log.info("Disconnecting robot.")
        try:
            robot.disconnect()
        except Exception:
            log.exception("robot.disconnect failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default=os.environ.get("SERVER_URL", "ws://127.0.0.1:8765"))
    p.add_argument("--task", default=os.environ.get("TASK", "fold clothing"))
    p.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "30")))
    p.add_argument("--chunk-reuse", type=int, default=20,
                   help="actions per chunk to execute before swapping in the next one")
    p.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "my_bimanual_follower"))
    p.add_argument("--calibration-dir",
                   default=os.environ.get(
                       "CALIBRATION_DIR",
                       str(Path(__file__).resolve().parents[2] / "calibration" / "openarm_follower")))
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
