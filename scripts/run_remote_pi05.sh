#!/usr/bin/env bash
# Drive the bimanual openarm with a remote pi05 policy.
#
# - Robot: real USB follower if LEFT_FOLLOWER + RIGHT_FOLLOWER are exported
#   (or --real-robot is passed); otherwise MuJoCo sim.
# - Cameras: BASE_CAMERA / LEFT_WRIST_CAMERA / RIGHT_WRIST_CAMERA are
#   auto-probed if not set (resolutions are baked in: wrist 1280x720, base 640x480).
# - Inference server URL: SERVER_URL, defaults to ws://localhost:8765.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MJPYTHON="${MJPYTHON:-${ROOT_DIR}/.venv/bin/python}"

[[ -x "$MJPYTHON" ]] || { echo "missing $MJPYTHON — set MJPYTHON=/path/to/mjpython"; exit 1; }

export SERVER_URL="${SERVER_URL:-ws://localhost:8765}"

exec "$MJPYTHON" "${ROOT_DIR}/app/run_remote_pi05.py" "$@"
