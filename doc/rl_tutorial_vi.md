# Tutorial: Reinforcement Learning trong Unitree RL Mjlab

Hướng dẫn này giải thích lý thuyết và cách implement các thuật toán RL trong repo, từ nền tảng đến chi tiết source code.

---

## Mục lục

1. [Tổng quan: RL là gì?](#1-tổng-quan-rl-là-gì)
2. [Bức tranh lớn: Các họ thuật toán RL](#2-bức-tranh-lớn-các-họ-thuật-toán-rl)
3. [Actor-Critic — Kiến trúc nền tảng](#3-actor-critic--kiến-trúc-nền-tảng)
4. [PPO — Vấn đề và giải pháp](#4-ppo--vấn-đề-và-giải-pháp)
5. [GAE — Ước lượng Advantage](#5-gae--ước-lượng-advantage)
6. [Rollout Storage — Bộ nhớ kinh nghiệm](#6-rollout-storage--bộ-nhớ-kinh-nghiệm)
7. [Training Loop — Vòng lặp huấn luyện](#7-training-loop--vòng-lặp-huấn-luyện)
8. [Adaptive Learning Rate](#8-adaptive-learning-rate)
9. [Multi-GPU Training](#9-multi-gpu-training)
10. [Kỹ thuật nâng cao](#10-kỹ-thuật-nâng-cao)
11. [Tổng kết Data Flow](#11-tổng-kết-data-flow)

---

## 1. Tổng quan: RL là gì?

### Vòng lặp cơ bản

```
Agent (Policy π) ──── action a_t ────→ Environment
       ↑                                     │
       └──── observation s_t, reward r_t ────┘
```

| Khái niệm | Trong robot locomotion |
|-----------|----------------------|
| **State s** | Góc khớp, vận tốc, IMU, ... |
| **Action a** | Lệnh vị trí/torque 23 khớp |
| **Reward r** | Đi đúng hướng = +, ngã = − |
| **Policy π(a\|s)** | Mạng neural: obs → action |
| **Episode** | Từ khi khởi động đến khi ngã hoặc hết giờ |

**Mục tiêu:** Tìm policy π* tối đa hóa tổng reward kỳ vọng:
```
J(π) = E[ Σ γ^t · r_t ]     với γ = 0.99 (discount factor)
```

`γ < 1` vì reward tương lai kém chắc chắn hơn reward hiện tại. `γ = 0.99` nghĩa là reward 100 bước sau có giá trị bằng `0.99^100 ≈ 37%` reward hiện tại.

---

## 2. Bức tranh lớn: Các họ thuật toán RL

Trước khi đi sâu vào PPO, cần hiểu nó nằm ở đâu trong bức tranh tổng thể.

### 2.1. Bản đồ các thuật toán RL

```
                        RL Algorithms
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    Model-Free          Model-Based      Hybrid
    (không cần mô          (học mô           │
     hình môi trường)    hình env)      (Dreamer, ...)
          │
    ┌─────┴──────┐
    │            │
 Value-Based  Policy-Based
 (học V/Q)    (học π trực tiếp)
    │            │
  DQN         REINFORCE
  C51          │
  Rainbow    Actor-Critic ← Chúng ta ở đây
               │
          ┌────┴────┐
          │         │
         A2C        PPO ← Thuật toán chính
         A3C       TRPO
                   SAC (off-policy AC)
```

### 2.2. So sánh các thuật toán phổ biến

| Thuật toán | Loại | Sample Efficiency | Ổn định | Phù hợp robot |
|-----------|------|------------------|---------|--------------|
| **DQN** | Value-based | Cao (replay buffer) | Trung bình | Không (discrete action) |
| **REINFORCE** | Policy gradient | Thấp | Thấp | Không |
| **A3C/A2C** | Actor-Critic | Trung bình | Trung bình | Được |
| **TRPO** | Trust region | Cao | Cao | Được nhưng phức tạp |
| **PPO** ✓ | Actor-Critic | Trung bình | **Cao** | **Tốt nhất** |
| **SAC** | Off-policy AC | **Cao** | Cao | Tốt cho continuous control |

**Tại sao PPO được chọn cho robot locomotion?**
- Ổn định với reward sparse/dense
- Parallel training trên hàng nghìn env đồng thời
- Code đơn giản, ít hyperparameter
- Hiệu quả với action space liên tục (continuous)
- Đã được validated rộng rãi (OpenAI, DeepMind, ETH Zurich)

---

## 3. Actor-Critic — Kiến trúc nền tảng

**File:** `mjlab/rsl_rl/modules/actor_critic.py`

### 3.1. Vấn đề của Pure Policy Gradient

**REINFORCE** (thuật toán policy gradient đơn giản nhất) cập nhật:
```
∇J(π) = E[ ∇log π(a|s) · G_t ]
```
Trong đó `G_t = r_t + r_{t+1} + ...` là tổng reward từ bước t.

**Vấn đề:** `G_t` có **variance rất cao** — cùng một hành động trong cùng trạng thái nhưng reward tổng lại khác nhau rất nhiều giữa các episode. Gradient noisy → training chậm, không ổn định.

**Giải pháp:** Thay `G_t` bằng **Advantage** `A(s,a) = Q(s,a) - V(s)`:
```
∇J(π) = E[ ∇log π(a|s) · A(s,a) ]
```
Advantage đo "hành động này tốt/xấu hơn mức trung bình *bao nhiêu*" — variance thấp hơn nhiều.

Nhưng để tính Advantage cần V(s) → cần thêm **Critic** để ước lượng V(s).

### 3.2. Kiến trúc Actor-Critic

```
                    ┌─────────────────────────────────────┐
                    │           Actor-Critic               │
                    │                                     │
  Observation ──→  │  ┌──────────────────────────────┐   │
  (IMU, joints,    │  │  Actor MLP (512→256→128)      │   │──→ action
   commands)       │  │  obs → mean, std → N(μ,σ)     │   │    (23 joints)
                   │  └──────────────────────────────┘   │
                   │                                     │
  Privileged  ──→  │  ┌──────────────────────────────┐   │
  Observation      │  │  Critic MLP (512→256→128)     │   │──→ V(s)
  (+ lin_vel)      │  │  obs → scalar value           │   │    (scalar)
                   │  └──────────────────────────────┘   │
                   └─────────────────────────────────────┘
```

```python
# actor_critic.py dòng 53-66
# Actor: observation → action distribution
self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)

# Critic: observation → value (scalar)
self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
```

### 3.3. Actor — Chọn hành động bằng phân phối Gaussian

Tại sao không output trực tiếp action mà lại output **phân phối**?

```
Training:  obs → actor → μ (mean)
                         σ (std, learnable parameter)
                         → N(μ, σ) → sample action  ← CÓ randomness → khám phá

Inference: obs → actor → μ (mean) → dùng trực tiếp  ← deterministic → ổn định
```

```python
# actor_critic.py dòng 118-151
def update_distribution(self, obs):
    mean = self.actor(obs)              # MLP forward pass
    std = self.std.expand_as(mean)      # σ: learnable parameter (ban đầu = 1.0)
    self.distribution = Normal(mean, std)

def act(self, obs, **kwargs):           # Training: sample
    obs = self.actor_obs_normalizer(obs)
    self.update_distribution(obs)
    return self.distribution.sample()  # a ~ N(μ, σ)

def act_inference(self, obs):          # Deploy: mean only
    obs = self.actor_obs_normalizer(obs)
    return self.actor(obs)             # return μ directly
```

**σ (std) là learnable parameter** — không phải cố định:
- Ban đầu `std = 1.0` → policy rất "rộng" → khám phá nhiều hành động khác nhau
- Trong quá trình training, PPO sẽ tự điều chỉnh std về mức tối ưu
- Std nhỏ dần → policy "tự tin" hơn → ít khám phá hơn

**Log-probability** — cần thiết cho PPO loss:
```python
def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)
    # log π(a|s) = Σ log N(a_i; μ_i, σ_i)   (mỗi joint độc lập)
```

### 3.4. Critic — Đánh giá trạng thái

```python
# actor_critic.py dòng 153-156
def evaluate(self, obs, **kwargs):
    obs = self.critic_obs_normalizer(obs)
    return self.critic(obs)   # V(s): "từ trạng thái này, kỳ vọng nhận được bao nhiêu reward?"
```

**Thông tin asymmetry — Actor vs Critic:**

| | Actor (deploy) | Critic (training only) |
|--|---------------|----------------------|
| IMU angular velocity | ✓ | ✓ |
| Projected gravity | ✓ | ✓ |
| Joint pos/vel | ✓ | ✓ |
| Commands | ✓ | ✓ |
| **Linear velocity (thực)** | ✗ | ✓ |

Critic có thêm thông tin "privileged" (không có trên robot thật) → ước lượng V(s) chính xác hơn → advantage chính xác hơn → training tốt hơn. Khi deploy chỉ cần Actor.

### 3.5. Observation Normalization

```python
# actor_critic.py dòng 58-72
if actor_obs_normalization:
    self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
```

**Vấn đề:** Observations có scale rất khác nhau:
- Góc khớp: `[-3, 3]` rad
- Vận tốc góc: `[-10, 10]` rad/s
- Gravity vector: `[-1, 1]`
- Commands: `[-2, 2]` m/s

Nếu không normalize, gradient từ observation scale lớn sẽ áp đảo.

**Giải pháp:** Running normalization — online update mean/variance:
```
obs_normalized = (obs - running_mean) / (running_std + ε)
```
Normalizer tự học từ dữ liệu, không cần biết trước min/max.

### 3.6. Kiến trúc MLP mặc định (G1)

```
Actor:
  input(~50) → Linear(50,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,23)
                                                                                         └─ 23 joint targets

Critic:
  input(~53) → Linear(53,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,1)
                                                                                         └─ scalar V(s)
```

**Tại sao ELU thay vì ReLU?**
- ReLU: `max(0, x)` — gradient = 0 với x < 0 (dying ReLU)
- ELU: `x nếu x>0, α(e^x - 1) nếu x<0` — gradient không bao giờ = 0
- Smooth hơn, converge nhanh hơn cho continuous control

---

## 4. PPO — Vấn đề và giải pháp

**File:** `mjlab/rsl_rl/algorithms/ppo.py`

### 4.1. Vấn đề PPO giải quyết

**Policy Gradient cơ bản** (REINFORCE, A2C):
```python
# Cập nhật policy đơn giản
loss = -log_prob * advantage
optimizer.step()
```

**Vấn đề 1: Bước update quá lớn có thể phá hỏng policy**

```
Policy tốt (iteration 1000)
    → Update lớn
    → Policy xấu đi nghiêm trọng
    → Khó recover vì dữ liệu mới thu thập bởi policy tệ
    → Diverge
```

Policy gradient không có giới hạn "bước nhảy" — một gradient step xấu có thể destroy mọi thứ đã học.

**Vấn đề 2: Sample inefficiency với vanilla policy gradient**

Mỗi batch dữ liệu chỉ dùng được **1 lần** để update (on-policy). Nếu muốn update nhiều lần trên cùng dữ liệu → distribution shift → gradient không còn valid.

### 4.2. Giải pháp của TRPO (tiền thân PPO)

TRPO (Trust Region Policy Optimization) giải quyết bằng **constraint tường minh**:
```
maximize:  E[ ratio · A ]
subject to: E[ KL(π_old || π_new) ] ≤ δ
```

Tốt về lý thuyết nhưng **quá phức tạp**: cần tính natural gradient, conjugate gradient solver, line search — khó implement, chậm.

### 4.3. Giải pháp của PPO: Clipping đơn giản, hiệu quả

PPO thay constraint của TRPO bằng **clipping trong objective function**:

```python
# ppo.py dòng 297-302
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
# ratio = π_new(a|s) / π_old(a|s)

surrogate        = -advantages_batch * ratio
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1-ε, 1+ε)

surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

**Visualize clipping:**

```
Khi Advantage > 0 (hành động TỐT, muốn tăng probability):

  -A·ratio
     │
     │  ╲
     │   ╲__________  ← clip: không cho ratio tăng quá 1+ε
     │              ╲
     └──────────────────→ ratio
        1-ε    1   1+ε

Khi Advantage < 0 (hành động XẤU, muốn giảm probability):

  -A·ratio  (A âm → đường dốc ngược)
     │
     │  __________
     │ ╱           ← clip: không cho ratio giảm quá 1-ε
     │╱
     └──────────────────→ ratio
        1-ε    1   1+ε
```

`torch.max(surrogate, surrogate_clipped)` = lấy giá trị **bi quan hơn** (pessimistic bound):
- Clip ngăn gradient đẩy policy ra ngoài "trust region"
- Không cần giải optimization có constraint phức tạp như TRPO

### 4.4. Tại sao có thể update nhiều lần trên cùng dữ liệu?

```
Thu thập 98,304 transitions bằng π_old
                    │
                    ▼
    Update π_new: epoch 1, batch 1  → ratio gần 1.0, OK
    Update π_new: epoch 1, batch 2  → ratio hơi xa 1.0, clip bắt đầu
    ...
    Update π_new: epoch 5, batch 4  → ratio bị clip nhiều → gradient nhỏ dần
                                                              → update tự dừng
```

Clipping **tự động kiểm soát** số lần update có ích — khi policy đã quá khác π_old, gradient bị clip → update không còn tác dụng → an toàn.

### 4.5. Value Function Loss

```python
# ppo.py dòng 305-313
if self.use_clipped_value_loss:
    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
        -self.clip_param, self.clip_param
    )
    value_losses         = (value_batch - returns_batch).pow(2)
    value_losses_clipped = (value_clipped - returns_batch).pow(2)
    value_loss = torch.max(value_losses, value_losses_clipped).mean()
else:
    value_loss = (returns_batch - value_batch).pow(2).mean()
```

Critic học để dự đoán `V(s)` bằng supervised learning:
- **Target**: `returns_batch` = GAE returns (R_t)
- **Prediction**: `value_batch` = V(s) hiện tại
- **Loss**: MSE = (prediction - target)²

Clipped value loss: Tương tự actor — giới hạn critic không thay đổi quá nhanh.

### 4.6. Entropy Bonus — Khuyến khích khám phá

```python
# ppo.py dòng 315
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()
```

**Entropy của Gaussian:** `H(N(μ,σ)) = 0.5 * log(2πeσ²)`

- σ lớn → entropy cao → policy đa dạng → nhiều khám phá
- Trừ entropy vào loss = **thưởng** cho policy có entropy cao
- `entropy_coef = 0.01` nhỏ → chỉ khuyến khích nhẹ, không làm policy quá random

### 4.7. Tổng Loss

```python
L_total = L_surrogate + c₁·L_value - c₂·H(π)
        =    (actor)  +  (critic)  +  (exploration)
```

| Thành phần | Mục đích | Hệ số |
|-----------|---------|-------|
| `L_surrogate` | Tối ưu hóa policy, có clipping | 1.0 |
| `c₁·L_value` | Huấn luyện critic | `value_loss_coef` = 1.0 |
| `-c₂·H(π)` | Duy trì entropy, tránh convergence sớm | `entropy_coef` = 0.01 |

### 4.8. Ưu và nhược điểm của PPO

**Ưu điểm:**
- **Đơn giản**: Không cần second-order optimization như TRPO
- **Ổn định**: Clipping ngăn update phá hủy policy
- **Sample reuse**: Dùng lại dữ liệu qua nhiều epoch một cách an toàn
- **Parallel-friendly**: Tự nhiên scale với nhiều env đồng thời
- **Robust**: Ít nhạy cảm với hyperparameter hơn nhiều thuật toán khác

**Nhược điểm:**
- **On-policy**: Mỗi rollout chỉ dùng một lần rồi bỏ → kém sample efficient hơn SAC
- **Sensitive to reward scale**: Reward quá lớn/nhỏ ảnh hưởng advantage estimation
- **Không tối ưu về sample efficiency**: Với cùng lượng data, SAC học tốt hơn trong nhiều bài toán
- **Khó tune cho continuous action**: std schedule, clip_param ảnh hưởng nhiều

**So sánh với SAC (Soft Actor-Critic):**

| | PPO | SAC |
|--|-----|-----|
| Kiểu | On-policy | Off-policy |
| Sample efficiency | Trung bình | Cao |
| Replay buffer | Không | Có |
| Entropy | Bonus heuristic | Tường minh trong objective |
| Cho robotics sim2real | Tốt | Tốt |
| Multi-env parallel | Rất tốt | Phức tạp hơn |
| Chọn khi nào | Sim với nhiều envs | Sim với ít envs hoặc real robot |

---

## 5. GAE — Ước lượng Advantage

**File:** `mjlab/rsl_rl/storage/rollout_storage.py` dòng 127-149

### 5.1. Vấn đề: Bias-Variance Tradeoff

Để tính Advantage `A(s,a) = Q(s,a) - V(s)` cần ước lượng Q(s,a). Có 2 cách cực đoan:

**Cách 1 — Monte Carlo (λ=1):**
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... (đến cuối episode)
A_t = G_t - V(s_t)
```
- **Bias = 0**: Không có sai số hệ thống (unbiased)
- **Variance cao**: Phụ thuộc vào toàn bộ chuỗi reward ngẫu nhiên
- Cần episode dài → chậm

**Cách 2 — 1-step TD (λ=0):**
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t
```
- **Variance thấp**: Chỉ dùng 1 bước
- **Bias cao**: Phụ thuộc vào độ chính xác của V(s) (bootstrapping bias)

**GAE** (λ=0.95): Tổ hợp có trọng số của tất cả n-step estimates:
```
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

### 5.2. Implement trong source code

```python
# rollout_storage.py dòng 127-149
def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):  # Duyệt ngược từ T→0

        next_values = last_values if step == T-1 else self.values[step + 1]

        # 1 nếu không phải terminal state
        next_is_not_terminal = 1.0 - self.dones[step].float()

        # TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]

        # GAE: A_t = δ_t + γλ·A_{t+1}  (đệ quy ngược)
        advantage = delta + next_is_not_terminal * gamma * lam * advantage

        # Return: R_t = A_t + V(s_t)
        self.returns[step] = advantage + self.values[step]

    self.advantages = self.returns - self.values

    if normalize_advantage:
        # Normalize: mean=0, std=1
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

**Tại sao duyệt ngược (reversed)?**

GAE là đệ quy: `A_t` cần `A_{t+1}`, `A_{t+1}` cần `A_{t+2}`, ... Duyệt ngược từ T về 0 mới có giá trị `A_{t+1}` khi tính `A_t`.

**Tại sao normalize advantages?**

Advantages có thể có giá trị rất khác nhau giữa các episodes. Normalize về mean=0, std=1:
- Gradient ổn định hơn
- Tránh một vài transitions có advantage cực lớn dominate update

### 5.3. Bootstrapping on Timeout

```python
# ppo.py dòng 160-163
if "time_outs" in extras:
    self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1), 1
    )
```

Phân biệt 2 kiểu kết thúc episode:
- **Termination** (ngã, va chạm): Robot thực sự thất bại → V(s_terminal) = 0
- **Truncation** (hết 20s): Robot vẫn đang hoạt động, chỉ bị cắt giữa chừng → V(s) ≠ 0

Nếu không xử lý truncation, policy sẽ học "trạng thái cuối episode luôn xấu" → bias nghiêm trọng.

Giải pháp: Thêm `γ·V(s)` vào reward cuối khi timeout → policy thấy "nếu tiếp tục sẽ còn nhận thêm reward".

---

## 6. Rollout Storage — Bộ nhớ kinh nghiệm

**File:** `mjlab/rsl_rl/storage/rollout_storage.py`

### 6.1. Cấu trúc buffer

Buffer lưu `T × N` transitions trước khi update (T=24 steps, N=4096 envs):

```python
# rollout_storage.py dòng 31-68
self.observations    = TensorDict(...)           # [T, N, obs_dim]
self.actions         = torch.zeros(T, N, act)    # [T, N, 23]
self.rewards         = torch.zeros(T, N, 1)      # [T, N, 1]
self.dones           = torch.zeros(T, N, 1)      # [T, N, 1]
self.values          = torch.zeros(T, N, 1)      # [T, N, 1]   ← V(s_t)
self.actions_log_prob = torch.zeros(T, N, 1)     # [T, N, 1]   ← log π(a|s)
self.mu              = torch.zeros(T, N, act)    # [T, N, 23]  ← action mean
self.sigma           = torch.zeros(T, N, act)    # [T, N, 23]  ← action std
self.returns         = torch.zeros(T, N, 1)      # [T, N, 1]   ← R_t (sau GAE)
self.advantages      = torch.zeros(T, N, 1)      # [T, N, 1]   ← A_t (sau GAE)
```

### 6.2. Mini-batch Generator — Tại sao shuffle?

```python
# rollout_storage.py dòng 160-203
def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    batch_size = N * T  # 4096 * 24 = 98,304

    # Shuffle ngẫu nhiên toàn bộ buffer
    indices = torch.randperm(batch_size)

    for epoch in range(num_epochs):      # 5 epochs
        for i in range(num_mini_batches): # 4 mini-batches
            batch_idx = indices[start:end]   # 24,576 samples
            yield obs[batch_idx], actions[batch_idx], ...
```

**Tại sao shuffle?** Các transitions liên tiếp trong thời gian có correlation cao (s_t và s_{t+1} rất giống nhau). Nếu train theo thứ tự thời gian, gradient sẽ bị bias. Shuffle phá vỡ temporal correlation → gradient đa dạng hơn.

---

## 7. Training Loop — Vòng lặp huấn luyện

**File:** `mjlab/rsl_rl/runners/on_policy_runner.py` dòng 62-176

### 7.1. Vòng lặp chính

```python
def learn(self, num_learning_iterations):
    obs = self.env.get_observations()

    for it in range(num_learning_iterations):  # 10,001 iterations

        # ═══ PHASE 1: ROLLOUT ═══════════════════════════════
        with torch.inference_mode():            # Không tính gradient
            for _ in range(24):                 # 24 steps

                actions = self.alg.act(obs)                          # π(a|s)
                obs, rewards, dones, extras = self.env.step(actions) # env step
                self.alg.process_env_step(obs, rewards, dones, extras) # store

            self.alg.compute_returns(obs)       # GAE

        # ═══ PHASE 2: UPDATE ════════════════════════════════
        loss_dict = self.alg.update()           # 5 epochs × 4 batches = 20 steps

        # ═══ PHASE 3: LOG & SAVE ════════════════════════════
        if it % 100 == 0:
            self.save(f"model_{it}.pt")         # .pt + .onnx
```

### 7.2. Phase 1: `alg.act()` — Thu thập dữ liệu

```python
# ppo.py dòng 129-140
def act(self, obs):
    self.transition.actions          = self.policy.act(obs).detach()
    self.transition.values           = self.policy.evaluate(obs).detach()   # V(s_t)
    self.transition.actions_log_prob = self.policy.get_actions_log_prob(...).detach()  # log π_old(a|s)
    self.transition.action_mean      = self.policy.action_mean.detach()    # μ (cần cho KL)
    self.transition.action_sigma     = self.policy.action_std.detach()     # σ (cần cho KL)
    self.transition.observations     = obs
    return self.transition.actions
```

`detach()` — **Quan trọng**: Tách tensor khỏi computation graph. Rollout phase không cần gradient, chỉ cần giá trị số để lưu vào buffer.

### 7.3. Phase 2: `alg.update()` — Cập nhật policy

```python
# ppo.py dòng 178-422 (simplified)
def update(self):
    generator = self.storage.mini_batch_generator(4, 5)  # 4 batches, 5 epochs

    for obs_batch, actions_batch, values_batch, advantages_batch, returns_batch,
        old_log_prob_batch, old_mu_batch, old_sigma_batch, ... in generator:

        # ── 1. Forward pass với policy MỚI ──────────────────
        self.policy.act(obs_batch)
        new_log_prob  = self.policy.get_actions_log_prob(actions_batch)
        new_value     = self.policy.evaluate(obs_batch)
        entropy       = self.policy.entropy

        # ── 2. KL → Adaptive LR ─────────────────────────────
        # (xem section 8)

        # ── 3. PPO Surrogate Loss ────────────────────────────
        ratio            = torch.exp(new_log_prob - old_log_prob_batch)
        surrogate        = -advantages_batch * ratio
        surrogate_clipped = -advantages_batch * torch.clamp(ratio, 0.8, 1.2)
        surrogate_loss   = torch.max(surrogate, surrogate_clipped).mean()

        # ── 4. Value Loss ────────────────────────────────────
        value_loss = (new_value - returns_batch).pow(2).mean()

        # ── 5. Total Loss ────────────────────────────────────
        loss = surrogate_loss + 1.0 * value_loss - 0.01 * entropy.mean()

        # ── 6. Gradient step ─────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

    self.storage.clear()
```

### 7.4. Tóm tắt số liệu 1 iteration

```
1 Iteration (G1 velocity):
├── Rollout:
│   ├── 24 steps × 4096 envs = 98,304 transitions
│   ├── Thời gian sim: 24 × 0.02s = 0.48s robot time
│   └── Tính GAE advantage cho tất cả transitions
│
├── Update:
│   ├── 5 epochs × 4 mini-batches = 20 gradient steps
│   ├── Mini-batch size: 98,304 / 4 = 24,576
│   └── ~20 backward passes
│
└── Toàn bộ training:
    ├── 10,001 iterations × 98,304 = ~983M transitions
    └── 10,001 × 20 = 200,020 gradient steps
```

---

## 8. Adaptive Learning Rate

**File:** `mjlab/rsl_rl/algorithms/ppo.py` dòng 260-294

### 8.1. Vấn đề với fixed learning rate

- LR quá lớn → policy thay đổi nhanh, unstable, có thể diverge
- LR quá nhỏ → học chậm, lãng phí compute
- LR tốt lúc đầu có thể không tốt ở giai đoạn sau

### 8.2. Giải pháp: Điều chỉnh theo KL divergence

KL divergence đo "khoảng cách" giữa 2 phân phối (policy mới vs cũ):

```python
# ppo.py dòng 262-269
kl = torch.sum(
    torch.log(sigma_new / sigma_old + 1e-5)         # log(σ_new/σ_old)
    + (sigma_old**2 + (mu_old - mu_new)**2)          # (σ_old² + (μ_old-μ_new)²)
      / (2 * sigma_new**2)                           # ÷ 2σ_new²
    - 0.5,
    axis=-1,
)
# Đây là KL(N(μ_old,σ_old) || N(μ_new,σ_new)) — công thức closed-form cho Gaussian
```

```python
# ppo.py dòng 280-284
if kl_mean > desired_kl * 2.0:          # Policy thay đổi QUÁ NHIỀU
    lr = max(1e-5, lr / 1.5)            # → Giảm LR 1.5x
elif kl_mean < desired_kl / 2.0:        # Policy thay đổi QUÁ ÍT
    lr = min(1e-2, lr * 1.5)            # → Tăng LR 1.5x
# Else: KL trong khoảng [0.005, 0.02] → giữ nguyên LR
```

Với `desired_kl = 0.01`:
```
KL > 0.02  → lr /= 1.5   (phanh lại)
KL < 0.005 → lr *= 1.5   (đạp ga thêm)
```

Cơ chế này hoạt động **song song với clipping** — clipping giới hạn theo từng sample, adaptive LR điều chỉnh learning speed toàn cục.

---

## 9. Multi-GPU Training

**File:** `mjlab/rsl_rl/algorithms/ppo.py` dòng 428-469

### 9.1. Kiến trúc

Mỗi GPU chạy một tập environments riêng, thu thập dữ liệu độc lập, rồi **đồng bộ gradient** trước khi update:

```
GPU 0 (4096 envs): rollout → gradients_0 ─┐
GPU 1 (4096 envs): rollout → gradients_1 ─┼─→ all_reduce (avg) ─→ update params
GPU 2 (4096 envs): rollout → gradients_2 ─┘

Kết quả: Mỗi GPU update với gradient trung bình trên 4096×3 = 12,288 envs
         Policy trên tất cả GPU là GIỐNG NHAU sau mỗi update
```

### 9.2. Broadcast parameters (1 lần trước training)

```python
# ppo.py dòng 428-439
def broadcast_parameters(self):
    model_params = [self.policy.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)  # GPU 0 → tất cả
    self.policy.load_state_dict(model_params[0])
```

### 9.3. Reduce gradients (mỗi gradient step)

```python
# ppo.py dòng 441-469
def reduce_parameters(self):
    grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
    all_grads = torch.cat(grads)                                      # Flatten
    torch.distributed.all_reduce(all_grads, op=ReduceOp.SUM)         # Cộng tất cả
    all_grads /= self.gpu_world_size                                   # Chia trung bình
    # Copy ngược lại vào param.grad
```

### 9.4. KL divergence trong multi-GPU

```python
# ppo.py dòng 272-275
if self.is_multi_gpu:
    torch.distributed.all_reduce(kl_mean, op=ReduceOp.SUM)
    kl_mean /= self.gpu_world_size
# → KL trung bình trên toàn bộ GPUs → adaptive LR nhất quán
```

---

## 10. Kỹ thuật nâng cao

### 10.1. RND — Random Network Distillation (Exploration bonus)

**Vấn đề:** Sparse reward — robot không nhận reward cho đến khi làm được việc khó. Policy gradient không thể học từ reward = 0 mãi.

**Giải pháp:** Tạo **intrinsic reward** dựa trên "sự mới lạ" của trạng thái:

```python
# ppo.py dòng 152-158
if self.rnd:
    intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
    self.transition.rewards += intrinsic_rewards
    # Total reward = extrinsic (env) + intrinsic (RND)
```

**Cơ chế:**
```
Target network  (frozen random MLP): obs → embedding_target   ← Không train
Predictor network (trained MLP):     obs → embedding_predict  ← Train để match target

Intrinsic reward = ||embedding_predict - embedding_target||²
```

- Trạng thái **đã thấy nhiều**: predictor đã học → prediction error thấp → reward thấp
- Trạng thái **mới lạ**: predictor chưa học → prediction error cao → reward cao → khuyến khích khám phá

### 10.2. Symmetry Augmentation

**Vấn đề:** Robot đi bộ có tính đối xứng trái-phải. Cần gấp đôi data để học cả 2 bên.

**Giải pháp:** Data augmentation bằng phép lật trái-phải:

```python
# ppo.py dòng 226-244
if self.symmetry["use_data_augmentation"]:
    obs_batch, actions_batch = data_augmentation_func(obs=obs_batch, actions=actions_batch)
    # obs được lật: joint trái ↔ phải, IMU sign flip
    # actions được lật: lệnh khớp trái ↔ phải
    # Batch size tăng gấp đôi: [B] → [2B]
```

Kết quả: Policy học nhanh gấp đôi vì mỗi bước đi bên trái cũng dạy cách đi bên phải.

### 10.3. Gradient Clipping

```python
# ppo.py dòng 380
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # max = 1.0
```

Nếu norm tổng của tất cả gradient > 1.0, scale xuống:
```
grad = grad × (1.0 / ||grad||)  khi ||grad|| > 1.0
```

Ngăn "gradient explosion" — thường xảy ra khi reward đột ngột rất lớn (vd: robot nhận penalty −200 khi ngã) → gradient lớn bất thường → params thay đổi đột ngột → network bị phá vỡ.

---

## 11. Tổng kết Data Flow

### Toàn bộ pipeline trong 1 iteration

```
┌──────────────────────────────────────────────────────────────┐
│                     ROLLOUT PHASE                            │
│                                                              │
│  for step in range(24):                                      │
│                                                              │
│    obs ──→ [Actor MLP] ──→ N(μ,σ) ──sample──→ action        │
│    obs ──→ [Critic MLP] ──→ V(s_t)                          │
│    log π(a|s) ← distribution.log_prob(action)               │
│                                                              │
│    action ──→ MuJoCo (4096 envs) ──→ obs', reward, done     │
│                                                              │
│    Store: (obs, action, reward, done, V, log_π, μ, σ)       │
│    RND: reward += ||predict(obs) - target(obs)||²           │
│    Timeout bootstrap: reward += γ·V(s) if timeout           │
│                                                              │
│  Compute GAE (duyệt ngược T→0):                             │
│    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)                       │
│    A_t = δ_t + γλ·A_{t+1}                                   │
│    R_t = A_t + V(s_t)                                       │
│    Normalize A: (A - mean) / std                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     UPDATE PHASE (×20)                       │
│                    [5 epochs × 4 batches]                    │
│                                                              │
│  Shuffle 98,304 transitions → 4 mini-batches (24,576 each)  │
│                                                              │
│  For each mini-batch:                                        │
│                                                              │
│    ┌─ ACTOR LOSS ──────────────────────────────────────┐     │
│    │ ratio = exp(log π_new - log π_old)                │     │
│    │ L_clip = -min(ratio·A, clip(ratio,0.8,1.2)·A)     │     │
│    └────────────────────────────────────────────────────┘     │
│                                                              │
│    ┌─ CRITIC LOSS ─────────────────────────────────────┐     │
│    │ L_value = (V(s) - R)²                             │     │
│    └────────────────────────────────────────────────────┘     │
│                                                              │
│    ┌─ ENTROPY BONUS ───────────────────────────────────┐     │
│    │ -H(π) = -0.5·log(2πe·σ²)                         │     │
│    └────────────────────────────────────────────────────┘     │
│                                                              │
│    L = L_clip + 1.0·L_value - 0.01·H(π)                     │
│    loss.backward() → clip_grad(1.0) → Adam.step()           │
│                                                              │
│    Adaptive LR: KL(π_new||π_old) vs desired_kl=0.01         │
│      KL > 0.02 → lr /= 1.5    KL < 0.005 → lr *= 1.5       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Hyperparameters mặc định (G1 velocity)

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `num_envs` | 4096 | Số robot chạy song song |
| `num_steps_per_env` | 24 | Bước rollout mỗi iteration |
| `max_iterations` | 10001 | Tổng số iterations |
| `gamma` | 0.99 | Discount (tầm nhìn ~100 bước) |
| `lam` | 0.95 | GAE lambda (bias/variance) |
| `clip_param` | 0.2 | PPO clipping ε |
| `learning_rate` | 1e-3 | LR ban đầu (adaptive sau) |
| `desired_kl` | 0.01 | Target KL cho adaptive LR |
| `num_learning_epochs` | 5 | Epochs mỗi update |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `entropy_coef` | 0.01 | Hệ số entropy bonus |
| `value_loss_coef` | 1.0 | Hệ số value loss |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `actor_hidden_dims` | (512, 256, 128) | Kiến trúc actor MLP |
| `critic_hidden_dims` | (512, 256, 128) | Kiến trúc critic MLP |
| `init_noise_std` | 1.0 | Noise ban đầu (exploration) |
| `activation` | ELU | Activation function |

### Throughput

```
1 iteration  = 24 × 4,096 = 98,304 transitions
10,001 iters = ~983 triệu transitions total
Per iter     = 20 gradient steps (5 epochs × 4 batches)
Total        = ~200,000 gradient steps
```