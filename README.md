# OpenArm Pi05 Hackathon

Remote inference and rapid adaptation experiments for a real bimanual OpenArm.

The robot stays on the local Linux machine with USB/CAN and cameras. The policy
runs on a remote GPU server over WebSocket. During the hackathon we validated
the inference loop, collected short OpenArm carrot-pick datasets, fine-tuned
Pi05 adapters, and started a lighter full ACT fine-tune.

## Demo

<video src="./demo.mp4" controls width="100%"></video>

[Watch the demo video](./demo.mp4)

## Participants

- Sichen Su (`SichenPa221`)
- Ziqi Ma
- Shangyu Yao
- Changqing Fu (`EvergreenTree`)

## System

- Robot: bimanual OpenArm follower.
- Local control: Linux robot host with CAN, cameras, and WebSocket client.
- Remote compute: GPU server at `ssh -p 17383 root@88.129.102.231`.
- Remote Python: `/root/venv/bin/python`.
- Original Pi05 checkpoint: `/root/checkpoints/ablation1-7_2`.
- WebSocket inference port: `8765`.

The local wrapper starts the remote server if needed and tunnels:

```text
localhost:8765 -> root@88.129.102.231:127.0.0.1:8765
```

See `AGENT.md` for server paths, ports, and debugging notes.

## Run The Robot

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

Set `TASK` to change the instruction:

```bash
TASK="pick up the carrot" ./scripts/run_remote_pi05_real.sh
```

Useful overrides:

```bash
# Server is already running.
START_REMOTE_SERVER=0 ./scripts/run_remote_pi05_real.sh

# A reachable server URL is already configured.
START_SSH_TUNNEL=0 SERVER_URL=ws://HOST:8765 ./scripts/run_remote_pi05_real.sh
```

## Data Notes

Datasets used in the final training round:

- `geoffroysu/openarm_carrot_30s_v2`
- `geoffroysu/openarm_carrot_30s_v3`
- `geoffroysu/openarm_carrot_30s_v4`

Important camera note:

- v2 had left/right wrist cameras reversed, so the prepared training dataset
  swapped them.
- v3 fixed the left/right wrist swap.
- v4 also uses the corrected left/right wrist mapping.

The prepared Pi05-format v4 dataset on the server is:

```text
/root/datasets/openarm_carrot_30s_v4_pi05
repo_id: cfu/openarm_carrot_30s_v4_pi05
episodes: 10
frames: 3000
fps: 15
state/action dim: 16
cameras: left_wrist, right_wrist, base
```

## Pi05 LoRA Runs

We fine-tuned the bimanual Pi05 checkpoint with LoRA on OpenArm carrot data.

### v2

- Dataset: `geoffroysu/openarm_carrot_30s_v2`
- Camera handling: swapped left/right wrist during preparation.
- Model repo: `cfu/openarm-carrot-30s-v2-pi05-lora`
- Purpose: first OpenArm carrot LoRA run with corrected dataset mapping.

### v3

- Dataset: `geoffroysu/openarm_carrot_30s_v3`
- Camera handling: no swap; v3 data fixed the camera labels.
- Model repo: `cfu/openarm-carrot-30s-v3-pi05-lora`
- Steps: 1200
- PEFT target: explicit attention-only LoRA, `q_proj,k_proj,v_proj,o_proj`
- Final loss: about `0.334`

This run proved the data path but only adapted self-attention, which is not
ideal when action/state heads need task-specific adjustment.

### v4

- Dataset: `geoffroysu/openarm_carrot_30s_v4`
- Camera handling: no swap.
- Model repo: `cfu/openarm-carrot-30s-v4-pi05-lora`
- Steps: 1200
- PEFT target: LeRobot Pi05 default LoRA target regex.
- Final loss: `0.445`

The saved adapter config confirms the default Pi05 target includes:

```text
gemma_expert self-attn q/v projections
model.state_proj
model.action_in_proj
model.action_out_proj
model.action_time_mlp_in
model.action_time_mlp_out
```

This is the preferred Pi05 LoRA setting for this project because it adapts the
Pi0.5 action path, including `action_out_proj`, instead of only attention.

## ACT Full Fine-Tune

To test a lighter policy, we started a full ACT fine-tune from:

```text
cfu/minerobots-srewdriver-act-20k
```

The source checkpoint is a 6-DOF single-arm ACT model. OpenArm v4 is 16-DOF
bimanual, so the action/state heads cannot be reused directly. We built a
compatibility initialization at:

```text
/root/checkpoints/minerobots-srewdriver-act-20k-openarm-v4-init
```

Loaded from the source checkpoint:

```text
229 / 234 tensors
```

Reinitialized because of the 6 -> 16 DOF mismatch:

```text
model.action_head.weight
model.action_head.bias
model.encoder_robot_state_input_proj.weight
model.vae_encoder_action_input_proj.weight
model.vae_encoder_robot_state_input_proj.weight
```

Training command shape:

```bash
lerobot-train \
  --dataset.repo_id=cfu/openarm_carrot_30s_v4_pi05 \
  --dataset.root=/root/datasets/openarm_carrot_30s_v4_pi05 \
  --steps=30000 \
  --policy.type=act \
  --policy.pretrained_path=/root/checkpoints/minerobots-srewdriver-act-20k-openarm-v4-init \
  --output_dir=/root/outputs/openarm_carrot_30s_v4_act_full \
  --job_name=openarm-carrot-30s-v4-act-full \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=cfu/openarm-carrot-30s-v4-act-full \
  --batch_size=8 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=1000 \
  --eval_freq=0 \
  --wandb.enable=false
```

Current status at wrap-up:

```text
repo: cfu/openarm-carrot-30s-v4-act-full
step: ~6500 / 30000
loss: ~0.089
speed: ~2.0 steps/s
GPU: ~74 GB / 98 GB VRAM
latest uploaded checkpoint: 006000
```

The ACT curve dropped quickly:

```text
step 100   loss 0.611
step 300   loss 0.330
step 700   loss 0.224
step 3000  loss ~0.12
step 6500  loss ~0.089
```

## Hugging Face Outputs

Final model repos:

- `cfu/openarm-carrot-30s-v2-pi05-lora`
- `cfu/openarm-carrot-30s-v3-pi05-lora`
- `cfu/openarm-carrot-30s-v4-pi05-lora`
- `cfu/openarm-carrot-30s-v4-act-full`

Source/reference model repos:

- `cfu/minerobots-srewdriver-act-20k`
- `cfu/minerobots-ball-act-100k`

## Lessons

- Camera label correctness matters as much as model choice. v2 required a
  wrist-camera swap; v3/v4 fixed it at data collection time.
- Pi05 LoRA should use LeRobot's Pi05 default PEFT targets for this setup,
  because they include the action/state projection path and `action_out_proj`.
- Full ACT fine-tuning is much lighter than Pi05 and runs at roughly
  `2 steps/s` with batch 8 on the remote GPU.
- When transferring ACT from a different DOF embodiment, reinitialize the
  mismatched state/action projection layers and keep only same-shaped trunk
  weights.
- Loss alone is not a rollout success metric. The next useful validation is
  real robot rollout success on carrot pick/place with the v4 Pi05 adapter and
  the ACT checkpoint.

## Local Robot Defaults

- `LEFT_FOLLOWER=can0`
- `RIGHT_FOLLOWER=can1`
- `BASE_CAMERA=0` at `640x480`
- `LEFT_WRIST_CAMERA=1` at `1280x720`
- `RIGHT_WRIST_CAMERA=2` at `1280x720`
- Default FPS: `30`

If all motors miss handshake, check:

1. 24V power and e-stop.
2. CAN H/L/GND wiring.
3. `can0` and `can1` ordering.
4. CAN FD setup: nominal `1 Mbps`, data `5 Mbps`.
5. Camera indices after reconnecting USB devices.

Do not commit secrets, SSH keys, GitHub tokens, or passwords.
