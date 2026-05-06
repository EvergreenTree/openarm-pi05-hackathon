#!/usr/bin/env bash
# Drive the *real* bimanual openarm follower with the remote pi05 policy.
#
# Uses local CAN/USB devices for the arms and cameras, and runs pi05 inference on the
# SSH GPU host via a local WebSocket tunnel. Override cameras with BASE_CAMERA /
# LEFT_WRIST_CAMERA / RIGHT_WRIST_CAMERA. Arm ports come from LEFT_FOLLOWER /
# RIGHT_FOLLOWER, defaulting to Linux SocketCAN can0/can1.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TASK="${TASK:-fold clothing}"

SSH_TARGET="${SSH_TARGET:-root@88.129.102.231}"
SSH_PORT="${SSH_PORT:-17383}"
REMOTE_SERVER_DIR="${REMOTE_SERVER_DIR:-/root}"
REMOTE_POLICY_DIR="${REMOTE_POLICY_DIR:-/root/checkpoints/ablation1-7_2}"
REMOTE_SERVER_PORT="${REMOTE_SERVER_PORT:-8765}"
LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-8765}"
START_REMOTE_SERVER="${START_REMOTE_SERVER:-1}"
START_SSH_TUNNEL="${START_SSH_TUNNEL:-1}"

LEFT_OPENARM_PORT="${LEFT_OPENARM_PORT:-${LEFT_FOLLOWER:-can0}}"
RIGHT_OPENARM_PORT="${RIGHT_OPENARM_PORT:-${RIGHT_FOLLOWER:-can1}}"

if [[ -z "${LEFT_OPENARM_PORT}" || -z "${RIGHT_OPENARM_PORT}" ]]; then
  ARM_PORTS=()
  for port in /dev/cu.usbserial-* /dev/cu.usbmodem* /dev/ttyUSB* /dev/ttyACM*; do
    [[ -e "${port}" ]] && ARM_PORTS+=("${port}")
  done
  if (( ${#ARM_PORTS[@]} > 0 )); then
    IFS=$'\n' ARM_PORTS=($(printf '%s\n' "${ARM_PORTS[@]}" | sort))
    unset IFS
  fi
  if (( ${#ARM_PORTS[@]} >= 2 )); then
    LEFT_OPENARM_PORT="${LEFT_OPENARM_PORT:-${ARM_PORTS[0]}}"
    RIGHT_OPENARM_PORT="${RIGHT_OPENARM_PORT:-${ARM_PORTS[1]}}"
    echo "[run_remote_pi05_real.sh] auto-picked arm ports: left=${LEFT_OPENARM_PORT} right=${RIGHT_OPENARM_PORT}" >&2
    echo "[run_remote_pi05_real.sh] set LEFT_OPENARM_PORT/RIGHT_OPENARM_PORT to override ordering" >&2
  fi
fi

export LEFT_OPENARM_PORT RIGHT_OPENARM_PORT
export LEFT_FOLLOWER="${LEFT_FOLLOWER:-${LEFT_OPENARM_PORT}}"
export RIGHT_FOLLOWER="${RIGHT_FOLLOWER:-${RIGHT_OPENARM_PORT}}"
export BASE_CAMERA="${BASE_CAMERA:-0}"
export LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA:-1}"
export RIGHT_WRIST_CAMERA="${RIGHT_WRIST_CAMERA:-2}"
export CAN_INTERFACE="${CAN_INTERFACE:-auto}"

if [[ "${START_REMOTE_SERVER}" != "0" ]]; then
  echo "[run_remote_pi05_real.sh] ensuring remote CUDA server on ${SSH_TARGET}:${REMOTE_SERVER_PORT}" >&2
  ssh -p "${SSH_PORT}" "${SSH_TARGET}" \
    "if [ -f '${REMOTE_SERVER_DIR}/pi05_server.pid' ] && kill -0 \$(cat '${REMOTE_SERVER_DIR}/pi05_server.pid') 2>/dev/null; then exit 0; fi; cd '${REMOTE_SERVER_DIR}'; nohup '${REMOTE_SERVER_DIR}/venv/bin/python' '${REMOTE_SERVER_DIR}/server.py' --policy-dir '${REMOTE_POLICY_DIR}' --host 127.0.0.1 --port '${REMOTE_SERVER_PORT}' --device cuda --dtype bfloat16 > '${REMOTE_SERVER_DIR}/server.log' 2>&1 < /dev/null & echo \$! > '${REMOTE_SERVER_DIR}/pi05_server.pid'"

  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
    if ssh -p "${SSH_PORT}" "${SSH_TARGET}" "ss -ltn 2>/dev/null | grep -q ':${REMOTE_SERVER_PORT} '" >/dev/null; then
      break
    fi
    sleep 2
  done
fi

if [[ "${START_SSH_TUNNEL}" != "0" ]]; then
  if lsof -nP -iTCP:"${LOCAL_SERVER_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[run_remote_pi05_real.sh] reusing existing localhost:${LOCAL_SERVER_PORT} listener" >&2
  else
    echo "[run_remote_pi05_real.sh] opening tunnel localhost:${LOCAL_SERVER_PORT} -> ${SSH_TARGET}:${REMOTE_SERVER_PORT}" >&2
    ssh -fN -p "${SSH_PORT}" \
      -L "127.0.0.1:${LOCAL_SERVER_PORT}:127.0.0.1:${REMOTE_SERVER_PORT}" \
      "${SSH_TARGET}"
  fi
fi

export SERVER_URL="${SERVER_URL:-ws://127.0.0.1:${LOCAL_SERVER_PORT}}"
export MJPYTHON="${MJPYTHON:-${ROOT_DIR}/.venv/bin/python}"
echo "[run_remote_pi05_real.sh] server=${SERVER_URL}" >&2

exec "${ROOT_DIR}/scripts/run_remote_pi05.sh" --real-robot --no-viewer "$@"
