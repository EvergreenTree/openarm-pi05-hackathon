"""Remote inference server for the pi05 ablation1-7_2 policy.

Run on the GPU box:
    ~/venv/bin/python server.py --policy-dir ~/checkpoints/ablation1-7_2 --port 8765

Holds one warm policy on cuda:0 and serves a single client at a time.
Each request goes obs -> action_chunk (30 x 16). The client is responsible for
executing the chunk at the control rate.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch
import websockets

from protocol import (
    ACTION_DIM,
    IMAGE_KEYS,
    STATE_DIM,
    decode_request,
    encode_response,
    jpeg_decode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")


class PolicyRunner:
    def __init__(self, policy_dir: Path, device: str = "cuda", dtype: str = "bfloat16"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.device = torch.device(device)
        log.info("Loading policy config from %s ...", policy_dir)
        cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
        cfg.pretrained_path = str(policy_dir)
        cfg.device = device
        cfg.dtype = dtype
        # max-autotune compile is for training; turn off for serving to keep startup fast.
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False

        policy_cls = get_policy_class(cfg.type)
        log.info("Building policy %s ...", cfg.type)
        self.policy = policy_cls.from_pretrained(str(policy_dir), config=cfg)
        self.policy.to(self.device).eval()

        # Pi05 was trained with QUANTILES normalization for state/action; the
        # processor files in the checkpoint contain the saved stats.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(policy_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        log.info("Policy ready on %s (dtype=%s).", self.device, dtype)

    def reset(self):
        self.policy.reset()

    @torch.inference_mode()
    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> np.ndarray:
        """Returns action chunk shape (chunk_size, ACTION_DIM) as numpy float32."""
        from lerobot.policies.utils import prepare_observation_for_inference

        # Mirror lerobot_record's input shape: numpy dict with `observation.state`
        # and `observation.images.*` (HWC uint8). The helper converts to tensors,
        # adds a batch dim, and moves to device.
        raw_obs = {"observation.state": state.astype(np.float32)}
        for name, arr in images.items():
            raw_obs[f"observation.images.{name}"] = arr  # HxWx3 uint8

        batch = prepare_observation_for_inference(raw_obs, self.device, task=task, robot_type="bi_openarm_follower")
        batch = self.preprocessor(batch)

        actions = self.policy.predict_action_chunk(batch)  # (1, chunk, padded_action_dim)
        actions = self.postprocessor(actions)             # unnormalize + abs (if relative)

        actions = actions.squeeze(0).float().cpu().numpy()  # (chunk, ACTION_DIM)
        assert actions.shape[-1] == ACTION_DIM, actions.shape
        return actions


def make_handler(runner: PolicyRunner):
    async def handler(ws):
        peer = ws.remote_address
        log.info("Client connected from %s", peer)
        runner.reset()
        try:
            async for msg in ws:
                t0 = time.perf_counter()
                state, jpeg_images, task = decode_request(msg)
                if state.shape != (STATE_DIM,):
                    raise ValueError(f"Bad state shape {state.shape}")
                images = {k: jpeg_decode(v) for k, v in jpeg_images.items()}
                missing = [k for k in IMAGE_KEYS if k not in images]
                if missing:
                    raise ValueError(f"Missing images: {missing}")

                t1 = time.perf_counter()
                actions = runner.predict_chunk(state, images, task)
                t2 = time.perf_counter()

                await ws.send(encode_response(actions, infer_ms=(t2 - t1) * 1000))
                log.info(
                    "chunk %s decode=%.1fms infer=%.1fms total=%.1fms",
                    actions.shape, (t1 - t0) * 1000, (t2 - t1) * 1000, (time.perf_counter() - t0) * 1000,
                )
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            log.exception("handler error: %s", e)
        finally:
            log.info("Client %s disconnected", peer)

    return handler


async def main_async(args):
    runner = PolicyRunner(Path(args.policy_dir), device=args.device, dtype=args.dtype)
    if args.warmup:
        log.info("Warmup forward pass...")
        dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
        dummy_imgs = {
            "left_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
            "right_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
            "base": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        for _ in range(2):
            t = time.perf_counter()
            runner.predict_chunk(dummy_state, dummy_imgs, "fold clothing")
            log.info("warmup step %.1f ms", (time.perf_counter() - t) * 1000)

    log.info("Listening on %s:%d", args.host, args.port)
    async with websockets.serve(
        make_handler(runner),
        args.host,
        args.port,
        max_size=32 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ):
        await asyncio.Future()  # run forever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-dir", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--warmup", action="store_true", default=True)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
