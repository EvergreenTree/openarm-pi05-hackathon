"""Drive a bimanual openarm with a remote pi05 policy.

Two robot backends:
  * MuJoCo sim (default): visualises with the passive viewer.
  * Real bimanual openarm follower: enabled with --real-robot, or implicitly
    when LEFT_FOLLOWER and RIGHT_FOLLOWER env vars are both set.

Cameras auto-probe by default. Override individually with env vars
BASE_CAMERA / LEFT_WRIST_CAMERA / RIGHT_WRIST_CAMERA.

Run:
    SERVER_URL=ws://localhost:8765 python app/run_remote_pi05.py
    SERVER_URL=ws://localhost:8765 python app/run_remote_pi05.py --real-robot

Auxiliary:
    python app/run_remote_pi05.py --list-cameras
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
# Reuse the wire format from the cloud-server side.
sys.path.insert(0, str(ROOT / "scripts" / "remote_inference"))
from protocol import IMAGE_KEYS, decode_response, encode_request, jpeg_encode  # noqa: E402

from cameras import CamSpec, ThreadedCamera, auto_assign, list_cameras  # noqa: E402
from mujoco_robot import MujocoBiOpenarm, MujocoBiOpenarmConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run")


# Per-stream resolutions the policy was trained with.
CAM_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "base":        (640, 480),
    "left_wrist":  (1280, 720),
    "right_wrist": (1280, 720),
}


def build_robot(args):
    if args.real_robot:
        from real_robot import RealBiOpenarm, RealBiOpenarmConfig, auto_find_arm_ports

        left, right = auto_find_arm_ports()
        if not (left and right):
            raise RuntimeError(
                "real_robot mode requested but USB arm ports not found. "
                "Set LEFT_FOLLOWER and RIGHT_FOLLOWER, or plug arms in."
            )
        return RealBiOpenarm(RealBiOpenarmConfig(
            left_port=left,
            right_port=right,
            calibration_dir=ROOT / "calibration" / "openarm_follower",
        ))
    return MujocoBiOpenarm(MujocoBiOpenarmConfig(mjcf_path=Path(args.mjcf)))


def _resolve_camera_indices() -> dict[str, int]:
    """Resolve which physical index serves each logical camera name.

    Use env vars when set; auto-probe (in detection order: base, L wrist,
    R wrist) for any missing slots.
    """
    env_map = {
        "base":        os.environ.get("BASE_CAMERA"),
        "left_wrist":  os.environ.get("LEFT_WRIST_CAMERA"),
        "right_wrist": os.environ.get("RIGHT_WRIST_CAMERA"),
    }
    if all(v is not None for v in env_map.values()):
        return {k: int(v) for k, v in env_map.items()}

    log.info("auto-probing cameras (env vars not all set)...")
    detected = auto_assign(["base", "left_wrist", "right_wrist"])
    out = {k: (int(env_map[k]) if env_map[k] is not None else detected[k]) for k in env_map}
    log.info("camera assignment: %s", out)
    return out


def build_cameras(args) -> dict[str, ThreadedCamera]:
    """{logical_name: capture}. Logical streams sharing one device share the
    underlying ThreadedCamera — macOS won't open the same index twice."""
    indices = _resolve_camera_indices()
    wanted: dict[str, tuple[int, int, int]] = {
        name: (indices[name], CAM_RESOLUTIONS[name][0], CAM_RESOLUTIONS[name][1])
        for name in CAM_RESOLUTIONS
    }
    by_index: dict[int, ThreadedCamera] = {}
    cams: dict[str, ThreadedCamera] = {}
    for name, (idx, w, h) in wanted.items():
        if idx in by_index:
            log.warning("camera index %d already opened — aliasing for %r", idx, name)
            cams[name] = by_index[idx]
            continue
        # If multiple logical streams share this device, request the largest
        # resolution any of them asked for so all consumers get enough pixels.
        sw = max(rw for n, (i, rw, _) in wanted.items() if i == idx)
        sh = max(rh for n, (i, _, rh) in wanted.items() if i == idx)
        cap = ThreadedCamera(CamSpec(name=f"cam{idx}", index=idx, width=sw, height=sh, fps=args.fps))
        by_index[idx] = cap
        cams[name] = cap
    return cams


async def control_loop(robot: MujocoBiOpenarm, cams, ws, viewer, task: str, fps: int, chunk_reuse: int):
    period = 1.0 / fps

    loop = asyncio.get_running_loop()

    def _encode(state_and_imgs):
        # Runs in a thread so it doesn't block the event loop / sim stepping.
        state, imgs = state_and_imgs
        # Model resizes to 224x224 internally; cap longest side at 320 for headroom.
        return state, {name: jpeg_encode(imgs[name], quality=80, max_dim=320) for name in IMAGE_KEYS}

    async def request_chunk():
        state = robot.get_state()
        imgs = {name: cams[name].read() for name in IMAGE_KEYS}
        state, jpegs = await loop.run_in_executor(None, _encode, (state, imgs))
        await ws.send(encode_request(state, jpegs, task))
        actions, infer_ms = decode_response(await ws.recv())
        return actions, infer_ms

    chunk, infer_ms = await request_chunk()
    log.info("first chunk shape=%s infer=%.1fms", chunk.shape, infer_ms)
    chunk_size = chunk.shape[0]
    chunk_reuse = min(chunk_reuse, chunk_size - 1)

    pending: asyncio.Task | None = None
    step_in_chunk = 0
    next_tick = time.perf_counter()

    while True:
        if viewer is not None and not viewer.is_running():
            log.info("viewer closed")
            break

        action_vec = chunk[step_in_chunk]
        robot.send_action(action_vec)
        robot.step(period)
        if viewer is not None:
            viewer.sync()

        step_in_chunk += 1

        if pending is None and step_in_chunk >= chunk_reuse // 2:
            pending = asyncio.create_task(request_chunk())

        if step_in_chunk >= chunk_reuse:
            if pending is None:
                pending = asyncio.create_task(request_chunk())
            t_wait = time.perf_counter()
            chunk, infer_ms = await pending
            pending = None
            step_in_chunk = 0
            log.info("swap chunk infer=%.1fms blocked=%.1fms",
                     infer_ms, (time.perf_counter() - t_wait) * 1000)

        next_tick += period
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            await asyncio.sleep(sleep)
        else:
            if sleep < -period:
                log.warning("loop fell behind by %.1f ms", -sleep * 1000)
            next_tick = time.perf_counter()


async def main_async(args):
    robot = build_robot(args)
    robot.connect()
    cams = build_cameras(args)

    viewer = None
    if args.viewer and robot.model is not None and robot.data is not None:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    try:
        log.info("connecting to %s", args.server_url)
        async with websockets.connect(
            args.server_url, max_size=32 * 1024 * 1024, ping_interval=20, ping_timeout=20,
        ) as ws:
            await control_loop(robot, cams, ws, viewer, args.task, args.fps, args.chunk_reuse)
    finally:
        for c in {id(c): c for c in cams.values()}.values():
            c.close()
        if viewer is not None:
            viewer.close()
        robot.disconnect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default=os.environ.get("SERVER_URL", "ws://127.0.0.1:8765"))
    p.add_argument("--task", default=os.environ.get("TASK", "fold clothing"))
    p.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "30")))
    p.add_argument("--chunk-reuse", type=int, default=25,
                   help="actions per chunk to execute before swapping in the prefetched next chunk")
    p.add_argument("--mjcf",
                   default=str(ROOT / "openarm_mujoco" / "v1" / "scene.xml"))
    p.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--list-cameras", action="store_true")
    # Real-robot mode: explicit --real-robot, OR implicit when both follower
    # ports are exported in the env (so the shell wrapper can auto-switch).
    real_default = bool(os.environ.get("LEFT_FOLLOWER") and os.environ.get("RIGHT_FOLLOWER"))
    p.add_argument("--real-robot", action=argparse.BooleanOptionalAction, default=real_default,
                   help="use the bimanual openarm USB follower instead of the MuJoCo sim")
    args = p.parse_args()

    if args.list_cameras:
        idxs = list_cameras()
        print("opened camera indices:", idxs)
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
