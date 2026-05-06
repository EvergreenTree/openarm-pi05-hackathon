# OpenArm Pi05 Hackathon

Remote Pi05 inference for a real bimanual OpenArm.

The robot stays on the local machine with USB/CAN and cameras. The policy runs
on a GPU server over WebSocket.

## Run

On the Linux robot machine:

```bash
python -m venv .venv
. .venv/bin/activate
pip install "lerobot[openarms,pi]==0.5.1" websockets opencv-python

lerobot-setup-can --mode=setup --interfaces=can0,can1

LEFT_FOLLOWER=can0 RIGHT_FOLLOWER=can1 \
BASE_CAMERA=0 LEFT_WRIST_CAMERA=1 RIGHT_WRIST_CAMERA=2 \
./scripts/run_remote_pi05_real.sh
```

Set `TASK` to change the instruction, for example:

```bash
TASK="fold clothing" ./scripts/run_remote_pi05_real.sh
```

See `AGENT.md` for server paths, ports, and debugging notes.
