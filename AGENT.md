# Agent Notes

This repo contains only the hackathon inference/control path.

## Architecture

- Local Linux robot machine: OpenArm followers, cameras, WebSocket client.
- Remote GPU server: Pi05 policy, WebSocket server.
- Wire format: `scripts/remote_inference/protocol.py`.

## GPU Server

- SSH target: `root@88.129.102.231`
- SSH port: `17383`
- Server directory: `/root`
- Server script: `/root/server.py`
- Python env: `/root/venv/bin/python`
- Policy checkpoint: `/root/checkpoints/ablation1-7_2`
- WebSocket port: `8765`
- PID file: `/root/pi05_server.pid`
- Log file: `/root/server.log`

The local wrapper starts the server if the PID file is stale, then tunnels:

```text
localhost:8765 -> root@88.129.102.231:127.0.0.1:8765
```

## Local Robot Defaults

- Left follower: `LEFT_FOLLOWER=can0`
- Right follower: `RIGHT_FOLLOWER=can1`
- Camera defaults:
  - `BASE_CAMERA=0` at `640x480`
  - `LEFT_WRIST_CAMERA=1` at `1280x720`
  - `RIGHT_WRIST_CAMERA=2` at `1280x720`
- Default task: `fold clothing`
- Default FPS: `30`

## Useful Commands

Set up Linux SocketCAN:

```bash
lerobot-setup-can --mode=setup --interfaces=can0,can1
```

Run the real robot:

```bash
LEFT_FOLLOWER=can0 RIGHT_FOLLOWER=can1 ./scripts/run_remote_pi05_real.sh
```

Check the remote server:

```bash
ssh -p 17383 root@88.129.102.231 'tail -f /root/server.log'
```

Skip server startup if it is already running:

```bash
START_REMOTE_SERVER=0 ./scripts/run_remote_pi05_real.sh
```

Skip SSH tunneling if `SERVER_URL` already points at a reachable server:

```bash
START_SSH_TUNNEL=0 SERVER_URL=ws://HOST:8765 ./scripts/run_remote_pi05_real.sh
```

## Debugging

If all motors miss handshake, check in this order:

1. 24V power and e-stop.
2. CAN H/L/GND wiring.
3. `can0` and `can1` ordering.
4. CAN FD setup: nominal `1 Mbps`, data `5 Mbps`.
5. Camera indices after reconnecting USB devices.

Do not commit secrets, SSH keys, GitHub tokens, or passwords.
