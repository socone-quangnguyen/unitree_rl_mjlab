# Hướng dẫn Custom Task & Action trong Unitree RL Mjlab

## Tổng quan kiến trúc

Mọi thứ trong framework này đều được cấu hình qua **dataclass config**. Bạn không cần sửa code core, chỉ cần:
1. Viết hàm MDP (reward/observation/termination)
2. Đăng ký trong config
3. Register task mới

```
Task Config = Scene + Actions + Observations + Rewards + Terminations + Commands + Events + Curriculum
```

---

## 1. Custom Reward (Phần hay custom nhất)

### Bước 1: Viết hàm reward

File: `mjlab/tasks/velocity/mdp/rewards.py`

```python
def my_custom_reward(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ví dụ: phạt robot nếu nghiêng quá threshold."""
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b shape: [num_envs, 3]
    tilt = torch.acos(-asset.data.projected_gravity_b[:, 2])
    return torch.where(tilt > threshold, -tilt, torch.zeros_like(tilt))
```

**Quy tắc:**
- Tham số đầu tiên luôn là `env: ManagerBasedRlEnv`
- Return shape: `[num_envs]` (1 giá trị per env)
- Dùng `env.scene["robot"]` để truy cập robot state

### Bước 2: Export trong `__init__.py`

File: `mjlab/tasks/velocity/mdp/__init__.py` — thêm:
```python
from .rewards import my_custom_reward
```

### Bước 3: Đăng ký trong env config

File: `mjlab/tasks/velocity/config/g1/env_cfgs.py`

```python
cfg.rewards["my_custom_reward"] = RewardTermCfg(
    func=mdp.my_custom_reward,
    weight=-2.0,           # Âm = phạt, Dương = thưởng
    params={
        "threshold": 0.3,
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

### Các reward có sẵn (tham khảo)

| Reward | Mục đích | Weight thường dùng |
|--------|----------|-------------------|
| `track_linear_velocity` | Theo dõi vận tốc tuyến tính | +1.0 |
| `track_angular_velocity` | Theo dõi vận tốc góc | +0.5 |
| `flat_orientation` | Giữ thân thẳng | -5.0 |
| `feet_air_time` | Thời gian chân trên không | +0.5 |
| `feet_slip` | Phạt trượt chân | -0.25 |
| `stand_still` | Đứng yên khi không có lệnh | -1.0 |
| `is_terminated` | Phạt khi ngã | -200.0 |
| `variable_posture` | Tư thế theo tốc độ | +1.0 |

---

## 2. Custom Observation

### Bước 1: Viết hàm observation

File: `mjlab/tasks/velocity/mdp/observations.py`

```python
def my_custom_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ví dụ: trả về chiều cao robot."""
    asset = env.scene[asset_cfg.name]
    # root_link_pos_w shape: [num_envs, 3]
    return asset.data.root_link_pos_w[:, 2:3]  # [num_envs, 1]
```

### Bước 2: Đăng ký trong config

File: `mjlab/tasks/velocity/velocity_env_cfg.py` (trong phần observations)

```python
# Thêm vào observation group "policy"
"my_obs": ObservationTermCfg(
    func=mdp.my_custom_obs,
    params={"asset_cfg": SceneEntityCfg("robot")},
    noise=Unoise(n_min=-0.01, n_max=0.01),  # Thêm noise cho sim2real
),
```

### Observations có sẵn

| Observation | Shape | Mô tả |
|-------------|-------|--------|
| `base_ang_vel` (IMU) | [3] | Vận tốc góc body |
| `projected_gravity` | [3] | Hướng trọng lực trong body frame |
| `command` | [3] | Lệnh vận tốc (vx, vy, wz) |
| `joint_pos_rel` | [num_joints] | Vị trí khớp (relative) |
| `joint_vel_rel` | [num_joints] | Vận tốc khớp |
| `last_action` | [num_actions] | Action bước trước |
| `foot_height` | [num_feet] | Chiều cao chân |
| `foot_contact` | [num_feet] | Trạng thái tiếp xúc |

### 2 nhóm observation

```python
observations = {
    "policy": ObservationGroupCfg(     # Policy network nhận
        enable_corruption=True,         # CÓ noise (giống real world)
        history_length=1,
    ),
    "critic": ObservationGroupCfg(     # Value network nhận
        enable_corruption=False,        # KHÔNG noise (privileged info)
    ),
}
```

---

## 3. Custom Action

### Đổi loại điều khiển

File: `mjlab/tasks/velocity/velocity_env_cfg.py`

```python
# Mặc định: Position control
actions = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),       # Tất cả actuator
        scale=0.5,                     # Scale action output
        use_default_offset=True,       # Offset bởi default pose
    ),
}

# Đổi sang Velocity control
actions = {
    "joint_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=1.0,
        use_default_offset=False,
    ),
}

# Đổi sang Torque control
actions = {
    "joint_effort": JointEffortActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=10.0,
    ),
}
```

### Custom scale per joint

```python
joint_pos_action.scale = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.3,
    r".*knee.*": 0.5,
    r".*ankle.*": 0.3,
}
```

---

## 4. Custom Termination

### Viết hàm termination

File: `mjlab/envs/mdp/terminations.py`

```python
def my_termination(
    env: ManagerBasedRlEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate nếu robot nhảy quá cao."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > max_height  # [num_envs] bool
```

### Đăng ký

```python
cfg.terminations["too_high"] = TerminationTermCfg(
    func=mdp.my_termination,
    params={"max_height": 2.0},
    # time_out=False  (mặc định) → đây là failure, không phải truncation
)
```

### Termination có sẵn

| Termination | Mô tả |
|-------------|--------|
| `time_out` | Hết thời gian episode (truncation) |
| `bad_orientation` | Nghiêng quá `limit_angle` |
| `root_height_below_minimum` | Rơi dưới chiều cao tối thiểu |
| `illegal_contact` | Va chạm không mong muốn |

---

## 5. Custom Command

### Thay đổi phạm vi vận tốc

```python
cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),  # Đổi lệnh mỗi 3-8 giây
    heading_command=True,
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),    # Tiến nhanh hơn
        lin_vel_y=(-0.8, 0.8),    # Đi ngang rộng hơn
        ang_vel_z=(-2.0, 2.0),    # Xoay nhanh hơn
        heading=(-math.pi, math.pi),
    ),
)
```

---

## 6. Custom Domain Randomization (Events)

### Thêm nhiễu mới

```python
# Đẩy robot ngẫu nhiên
cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",                    # Chạy định kỳ
    interval_range_s=(1.0, 3.0),       # Mỗi 1-3 giây
    params={
        "velocity_range": {
            "x": (-1.0, 1.0),          # Mạnh hơn mặc định
            "y": (-1.0, 1.0),
            "z": (-0.5, 0.5),
        },
    },
)

# Randomize ma sát mặt đất
cfg.events["body_friction"] = EventTermCfg(
    mode="startup",                     # Chỉ chạy 1 lần
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
    },
)
```

**Event modes:**
- `"startup"` — chạy 1 lần khi khởi tạo
- `"reset"` — chạy mỗi khi reset episode
- `"interval"` — chạy định kỳ (cần `interval_range_s`)

---

## 7. Custom Curriculum

### Tăng dần độ khó vận tốc

```python
cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
        "command_name": "twist",
        "velocity_stages": [
            # Ban đầu: chậm
            {"step": 0,
             "lin_vel_x": (-0.5, 1.0),
             "lin_vel_y": (-0.5, 0.5),
             "ang_vel_z": (-1.0, 1.0)},
            # Sau 5000 steps: nhanh hơn
            {"step": 5000 * 24,
             "lin_vel_x": (-1.0, 2.0),
             "lin_vel_y": (-1.0, 1.0)},
            # Sau 10000 steps: nhanh nhất
            {"step": 10000 * 24,
             "lin_vel_x": (-1.5, 3.0),
             "lin_vel_y": (-1.5, 1.5),
             "ang_vel_z": (-2.0, 2.0)},
        ],
    },
)
```

---

## 8. Tạo Task mới hoàn chỉnh (Ví dụ: G1 chạy nhanh)

### Bước 1: Tạo thư mục config

```
mjlab/tasks/velocity/config/g1_sprint/
├── __init__.py
├── env_cfgs.py
└── rl_cfg.py
```

### Bước 2: env_cfgs.py

```python
import math
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp import velocity_command as vc
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.asset_zoo.robots.unitree_g1 import get_g1_robot_cfg
# ... các import khác

def g1_sprint_env_cfg(play: bool = False):
    cfg = make_velocity_env_cfg()

    # Robot
    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    # Sensors (copy từ g1 config, hoặc tự define)
    # ...

    # Tăng phạm vi vận tốc
    cfg.commands["twist"].ranges.lin_vel_x = (0.5, 3.0)  # Chỉ tiến, nhanh
    cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)  # Ít đi ngang
    cfg.commands["twist"].ranges.ang_vel_z = (-0.5, 0.5)  # Ít xoay

    # Tăng reward cho velocity tracking
    cfg.rewards["track_linear_velocity"].weight = 2.0

    # Giảm phạt năng lượng (cho phép dùng nhiều lực hơn)
    cfg.rewards["action_rate_l2"].weight = -0.005

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False

    return cfg
```

### Bước 3: rl_cfg.py

```python
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

def g1_sprint_ppo_runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-3,
            num_learning_epochs=5,
            num_mini_batches=4,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
        ),
        experiment_name="g1_sprint",
        num_steps_per_env=24,
        max_iterations=10001,
        save_interval=100,
    )
```

### Bước 4: __init__.py (đăng ký task)

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import g1_sprint_env_cfg
from .rl_cfg import g1_sprint_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Sprint-Unitree-G1",
    env_cfg=g1_sprint_env_cfg(),
    play_env_cfg=g1_sprint_env_cfg(play=True),
    rl_cfg=g1_sprint_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

### Bước 5: Import task

File: `mjlab/tasks/velocity/config/__init__.py` — thêm:
```python
from . import g1_sprint
```

### Bước 6: Train

```bash
python scripts/train.py Mjlab-Velocity-Sprint-Unitree-G1 --env.scene.num-envs=4096
```

---

## 9. Override nhanh qua CLI (không cần sửa code)

```bash
# Đổi reward weight
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --env.rewards.track_linear_velocity.weight=2.0

# Đổi learning rate
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.algorithm.learning-rate=5e-4

# Đổi network architecture
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.policy.actor-hidden-dims="[256,128,64]"

# Đổi số env và iterations
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=5000

# Đổi episode length
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --env.episode-length-s=30.0
```

---

## 10. Dữ liệu Robot có thể truy cập

Trong mọi hàm reward/observation, bạn có thể truy cập:

```python
asset = env.scene["robot"]

# Vị trí & hướng
asset.data.root_link_pos_w          # [num_envs, 3] - vị trí world
asset.data.root_link_quat_w         # [num_envs, 4] - quaternion
asset.data.projected_gravity_b      # [num_envs, 3] - gravity trong body frame

# Vận tốc
asset.data.root_com_lin_vel_b       # [num_envs, 3] - vận tốc tuyến tính body
asset.data.root_com_ang_vel_b       # [num_envs, 3] - vận tốc góc body

# Khớp
asset.data.joint_pos                # [num_envs, num_joints] - vị trí khớp
asset.data.joint_vel                # [num_envs, num_joints] - vận tốc khớp
asset.data.default_joint_pos        # [num_envs, num_joints] - vị trí mặc định

# Body state
asset.data.body_pos_w               # [num_envs, num_bodies, 3]
asset.data.body_lin_vel_w            # [num_envs, num_bodies, 3]

# Sensors
sensor = env.scene["feet_ground_contact"]
sensor.data.force                    # [num_envs, num_feet, 3]
sensor.data.found                    # [num_envs, num_feet] bool
sensor.data.air_time                 # [num_envs, num_feet]

# Commands
cmd = env.command_manager.get_command("twist")  # [num_envs, 3]
```

---

## Tổng kết: Custom cái gì → sửa ở đâu

| Muốn custom | File cần sửa |
|-------------|-------------|
| Reward mới | `mjlab/tasks/velocity/mdp/rewards.py` + env config |
| Observation mới | `mjlab/tasks/velocity/mdp/observations.py` + env config |
| Action type | env config (`actions` dict) |
| Command range | env config (`commands` dict) |
| Termination | `mjlab/envs/mdp/terminations.py` + env config |
| Domain randomization | env config (`events` dict) |
| Curriculum | env config (`curriculum` dict) |
| Network architecture | rl config (`policy` section) |
| PPO hyperparams | rl config (`algorithm` section) |
| Task mới hoàn chỉnh | Tạo thư mục config mới + register |
| Nhanh không sửa code | CLI override (`--env.rewards...`) |