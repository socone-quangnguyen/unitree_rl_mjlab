# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Mjlab is a reinforcement learning framework for Unitree robot locomotion control. It uses MuJoCo (with GPU-accelerated MuJoCo-Warp) as the physics backend and an Isaac Lab-inspired API for modular RL training and sim-to-real deployment. Supported robots: Go2, G1, G1-23DOF, H1_2.

## Common Commands

```bash
# Install (Python 3.11 conda env required)
pip install -e .

# List all registered tasks
python scripts/list_envs.py

# Train velocity tracking (single GPU)
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096

# Train with multiple GPUs
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --gpu-ids 0 1 --env.scene.num-envs=4096

# Train motion imitation (requires --motion_file)
python scripts/train.py Mjlab-Tracking-Flat-Unitree-G1 --motion_file=mjlab/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096

# Play/visualize a trained policy
python scripts/play.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint_file=logs/rsl_rl/.../model_xx.pt

# Play with zero or random actions (no checkpoint needed)
python scripts/play.py Mjlab-Velocity-Flat-Unitree-G1 --agent=zero

# Convert motion CSV to NPZ
python scripts/csv_to_npz.py --input-file mjlab/motions/g1/file.csv --output-name file.npz --input-fps 30 --output-fps 50
```

All config parameters are overridable via CLI using tyro (e.g., `--env.scene.num-envs=4096`, `--agent.seed=42`, `--env.rewards`, `--agent.policy`).

## Architecture

### Manager-Based Environment System

The core is `ManagerBasedRlEnv` (`mjlab/envs/manager_based_rl_env.py`) which orchestrates modular managers:
- **ActionManager** — converts policy outputs to joint commands
- **ObservationManager** — computes observation tensors from simulation state
- **RewardManager** — calculates reward terms with configurable weights
- **TerminationManager** — determines episode termination conditions
- **CommandManager** — generates task targets (velocity commands or motion references)
- **CurriculumManager** — progressive difficulty scaling
- **EventManager** — domain randomization

All managers are configured via frozen dataclasses that compose into `ManagerBasedRlEnvCfg`.

### Task Registration

Tasks are registered at import time via `register_mjlab_task()` in `mjlab/tasks/registry.py`. Each robot+terrain combination has its own config directory:
- `mjlab/tasks/velocity/config/{go2,g1,g1_23dof,h1_2}/` — velocity tracking tasks
- `mjlab/tasks/tracking/config/{g1,...}/` — motion imitation tasks

Registration pattern (`__init__.py` in each config dir):
```python
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Unitree-G1",
    env_cfg=unitree_g1_flat_env_cfg(),
    play_env_cfg=unitree_g1_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

### Task Types

**Velocity tracking** (`mjlab/tasks/velocity/`): Robot tracks linear/angular velocity commands. MDP components in `mdp/` define rewards (velocity tracking, energy, stability), observations (IMU, joint state, commands), and commands.

**Motion imitation** (`mjlab/tasks/tracking/`): Robot imitates reference motion sequences from `.npz` files. Based on BeyondMimic. Rewards track body pose/velocity against reference frames.

### MDP Function Convention

Reward and observation functions follow this signature:
```python
def some_reward(env: ManagerBasedRlEnv, param1: float, ...) -> torch.Tensor:
    asset = env.scene["robot"]
    # compute and return tensor
```

### RL Training Pipeline

`mjlab/rsl_rl/` contains the training infrastructure:
- `runners/` — `OnPolicyRunner` (PPO rollouts + training loop)
- `algorithms/` — PPO implementation
- `networks/` — Actor-Critic policy networks
- `storage/` — Rollout buffer

Multi-GPU training uses `torchrunx`. Checkpoints saved to `logs/rsl_rl/{experiment_name}/{timestamp}/`.

### Simulation Layer

`mjlab/sim/` wraps MuJoCo with Warp GPU acceleration. `mjlab/scene/` manages entities and sensors. Robot assets (MJCF XML models, joint configs) live in `mjlab/asset_zoo/robots/`.

### Deployment

`deploy/robots/{go2,g1,g1_23dof,h1_2}/` contain C++ programs that load exported ONNX policies and control real robots via Unitree SDK. Build with CMake:
```bash
cd deploy/robots/g1 && mkdir build && cd build && cmake .. && make
./g1_ctrl --network=enp5s0
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `mjlab/envs/` | Core RL environment (`ManagerBasedRlEnv`) |
| `mjlab/managers/` | All manager implementations |
| `mjlab/tasks/` | Task configs, MDP components, registration |
| `mjlab/rsl_rl/` | PPO runner, algorithms, networks |
| `mjlab/sim/` | MuJoCo simulation wrapper |
| `mjlab/asset_zoo/` | Robot MJCF models and configs |
| `mjlab/sensor/` | Contact, IMU, raycast sensors |
| `mjlab/terrains/` | Terrain generation |
| `mjlab/viewer/` | Native MuJoCo and Viser web viewers |
| `scripts/` | Train, play, and utility entry points |
| `deploy/` | C++ real-robot deployment code |

## Configuration Hierarchy

Environment configs are built with factory functions (e.g., `unitree_g1_flat_env_cfg()`) that return `ManagerBasedRlEnvCfg` dataclasses. These compose: scene config, observation groups, action terms, reward terms with weights, termination conditions, curriculum stages, command generators, and MuJoCo simulation parameters. RL configs (`RslRlOnPolicyRunnerCfg`) set PPO hyperparameters and network architecture.

Training output: `logs/rsl_rl/{experiment_name}/{YYYY-MM-DD_HH-MM-SS}/` containing model checkpoints (`.pt`), exported ONNX policy, config snapshots (`params/env.yaml`, `params/agent.yaml`), and optional training videos.