"""Wire format shared by client and server.

Single websocket message per inference call. Payload is msgpack-encoded:

    request:  {"state": [16 floats], "images": {name: jpeg_bytes}, "task": str}
    response: {"actions": list[list[float]],  # shape (chunk_size, 16)
               "infer_ms": float}

Images are sent as JPEG bytes (already in HWC, RGB uint8). The server decodes,
converts to CHW float[0,1] tensors, and runs the policy.
"""
from __future__ import annotations

import io

import msgpack
import numpy as np

# Same names the policy was trained with — must match config.json input_features.
IMAGE_KEYS = ("left_wrist", "right_wrist", "base")
STATE_DIM = 16
ACTION_DIM = 16


def encode_request(state: np.ndarray, jpeg_images: dict[str, bytes], task: str) -> bytes:
    assert state.shape == (STATE_DIM,), state.shape
    return msgpack.packb(
        {
            "state": state.astype(np.float32).tolist(),
            "images": {k: jpeg_images[k] for k in IMAGE_KEYS},
            "task": task,
        },
        use_bin_type=True,
    )


def decode_request(buf: bytes):
    obj = msgpack.unpackb(buf, raw=False)
    state = np.asarray(obj["state"], dtype=np.float32)
    return state, obj["images"], obj["task"]


def encode_response(actions: np.ndarray, infer_ms: float) -> bytes:
    return msgpack.packb(
        {"actions": actions.astype(np.float32).tolist(), "infer_ms": float(infer_ms)},
        use_bin_type=True,
    )


def decode_response(buf: bytes):
    obj = msgpack.unpackb(buf, raw=False)
    return np.asarray(obj["actions"], dtype=np.float32), float(obj["infer_ms"])


def jpeg_encode(rgb_hwc_uint8: np.ndarray, quality: int = 85, max_dim: int | None = None) -> bytes:
    """RGB uint8 HxWx3 -> JPEG bytes. Uses cv2 (libjpeg-turbo) — much faster than PIL.

    If `max_dim` is set, the image is downscaled so its longest side equals
    `max_dim` before encoding. The pi05 model resizes everything to 224x224
    internally, so sending full HD is wasteful.
    """
    import cv2

    img = rgb_hwc_uint8
    if max_dim is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / m
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def jpeg_decode(data: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
