"""Threaded macOS camera capture for the policy's three image inputs.

Uses cv2.VideoCapture with the AVFoundation backend so Continuity Camera
iPhones (over USB) and the built-in face cam appear as plain integer indices.

Each camera runs in its own daemon thread; `read()` returns the most recent
RGB frame without blocking the control loop.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CamSpec:
    name: str
    index: int
    width: int
    height: int
    fps: int = 30


class ThreadedCamera:
    def __init__(self, spec: CamSpec):
        self.spec = spec
        self._cap = cv2.VideoCapture(spec.index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {spec.name}: failed to open index {spec.index}")
        # On macOS some properties only stick on the second set call.
        for _ in range(2):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            self._cap.set(cv2.CAP_PROP_FPS, spec.fps)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{spec.name}", daemon=True)
        self._thread.start()
        # Wait briefly for the first frame so callers can read immediately.
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.02)
        if self._frame is None:
            raise RuntimeError(f"camera {spec.name}: no frame within 5s")

    def _loop(self):
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb

    def read(self) -> np.ndarray:
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"camera {self.spec.name}: no frame")
            return self._frame.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


def list_cameras(max_index: int = 8) -> list[int]:
    """Return indices that opened successfully under AVFoundation."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
            cap.release()
    return found


def auto_assign(logical_names: list[str], max_probe: int = 8) -> dict[str, int]:
    """Map logical camera names to detected indices.

    Probes 0..max_probe-1; assigns names to indices in detection order.
    If fewer indices are detected than names requested, the last index is reused
    (aliased) for the trailing names — useful when only the laptop webcam is
    plugged in.
    """
    detected = list_cameras(max_probe)
    if not detected:
        raise RuntimeError("no cameras detected")
    mapping: dict[str, int] = {}
    for i, name in enumerate(logical_names):
        mapping[name] = detected[i] if i < len(detected) else detected[-1]
    return mapping
