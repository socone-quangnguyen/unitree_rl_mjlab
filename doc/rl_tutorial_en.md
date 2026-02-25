# Tutorial: Reinforcement Learning in Unitree RL Mjlab

This guide explains the theory and implementation of Reinforcement Learning (RL) algorithms in this repository, from fundamentals to source code details.

---

## Table of Contents

1. [Overview: What is RL?](#1-overview-what-is-rl)
2. [The Big Picture: RL Algorithm Families](#2-the-big-picture-rl-algorithm-families)
3. [Actor-Critic — Foundation Architecture](#3-actor-critic--foundation-architecture)
4. [PPO — Problems and Solutions](#4-ppo--problems-and-solutions)
5. [GAE — Advantage Estimation](#5-gae--advantage-estimation)
6. [Rollout Storage — Experience Memory](#6-rollout-storage--experience-memory)
7. [Training Loop](#7-training-loop)
8. [Adaptive Learning Rate](#8-adaptive-learning-rate)
9. [Multi-GPU Training](#9-multi-gpu-training)
10. [Advanced Techniques](#10-advanced-techniques)
11. [Data Flow Summary](#11-data-flow-summary)

---

## 1. Overview: What is RL?

### Basic Loop

```
Agent (Policy π) ──── action a_t ────→ Environment
       ↑                                     │
       └──── observation s_t, reward r_t ────┘
```

| Concept | In Robot Locomotion |
|-----------|----------------------|
| **State s** | Joint angles, velocity, IMU, ... |
| **Action a** | Position/torque commands for joints |
| **Reward r** | Moving in right direction = +, falling = − |
| **Policy π(a\|s)** | Neural Network: obs → action |
| **Episode** | From startup until falling or timeout |

**Goal:** Find an optimal policy π* that maximizes the expected total reward:
```
J(π) = E[ Σ γ^t · r_t ]     with γ = 0.99 (discount factor)
```

`γ < 1` because future rewards are less certain than current ones. `γ = 0.99` means a reward 100 steps in the future is worth `0.99^100 ≈ 37%` of a current reward.

---

## 2. The Big Picture: RL Algorithm Families

Before diving into PPO, it's important to understand where it sits in the overall landscape.

### 2.1. Map of RL Algorithms

```
                        RL Algorithms
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    Model-Free          Model-Based      Hybrid
    (No environment      (Learns env      │
     model needed)        model)         (Dreamer, ...)
          │
    ┌─────┴──────┐
    │            │
 Value-Based  Policy-Based
 (Learns V/Q) (Learns π directly)
    │            │
  DQN         REINFORCE
  C51          │
  Rainbow    Actor-Critic ← We are here
               │
          ┌────┴────┐
          │         │
         A2C        PPO ← Main Algorithm
         A3C       TRPO
                   SAC (off-policy AC)
```

### 2.2. Comparison of Popular Algorithms

| Algorithm | Type | Sample Efficiency | Stability | Robot Suitability |
|-----------|------|------------------|---------|--------------|
| **DQN** | Value-based | High (replay buffer) | Medium | No (discrete action) |
| **REINFORCE** | Policy gradient | Low | Low | No |
| **A3C/A2C** | Actor-Critic | Medium | Medium | Yes |
| **TRPO** | Trust region | High | High | Yes but complex |
| **PPO** ✓ | Actor-Critic | Medium | **High** | **Best** |
| **SAC** | Off-policy AC | **High** | High | Good for continuous control |

**Why PPO for robot locomotion?**
- Stable with sparse/dense rewards.
- Parallel training on thousands of environments simultaneously.
- Simple code, few hyperparameters.
- Effective with continuous action spaces.
- Widely validated (OpenAI, DeepMind, ETH Zurich).

---

## 3. Actor-Critic — Foundation Architecture

**File:** `mjlab/rsl_rl/modules/actor_critic.py`

### 3.1. Problems with Pure Policy Gradient

**REINFORCE** (simplest policy gradient algorithm) updates:
```
∇J(π) = E[ ∇log π(a|s) · G_t ]
```
Where `G_t = r_t + r_{t+1} + ...` is the total reward from step t.

**Problem:** `G_t` has **very high variance** — the same action in the same state can lead to very different total rewards across episodes. Noisy gradients → slow, unstable training.

**Solution:** Replace `G_t` with **Advantage** `A(s,a) = Q(s,a) - V(s)`:
```
∇J(π) = E[ ∇log π(a|s) · A(s,a) ]
```
Advantage measures "how much better/worse is this action than the average" — much lower variance.

However, to calculate Advantage we need V(s) → we need a **Critic** to estimate V(s).

### 3.2. Actor-Critic Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Actor-Critic               │
                    │                                     │
  Observation ──→  │  ┌──────────────────────────────┐   │
  (IMU, joints,    │  │  Actor MLP (512→256→128)      │   │──→ action
   commands)       │  │  obs → mean, std → N(μ,σ)     │   │    (joint targets)
                   │  └──────────────────────────────┘   │
                   │                                     │
  Privileged  ──→  │  ┌──────────────────────────────┐   │
  Observation      │  │  Critic MLP (512→256→128)     │   │──→ V(s)
  (+ lin_vel)      │  │  obs → scalar value           │   │    (scalar)
                   │  └──────────────────────────────┘   │
                   └─────────────────────────────────────┘
```

```python
# actor_critic.py lines 53-66
# Actor: observation → action distribution
self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)

# Critic: observation → value (scalar)
self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
```

### 3.3. Actor — Action Selection via Gaussian Distribution

Why output a **distribution** instead of the action directly?

```
Training:  obs → actor → μ (mean)
                         σ (std, learnable parameter)
                         → N(μ, σ) → sample action  ← HAS randomness → exploration

Inference: obs → actor → μ (mean) → used directly  ← deterministic → stability
```

```python
# actor_critic.py lines 118-151
def update_distribution(self, obs):
    mean = self.actor(obs)              # MLP forward pass
    std = self.std.expand_as(mean)      # σ: learnable parameter (initially 1.0)
    self.distribution = Normal(mean, std)

def act(self, obs, **kwargs):           # Training: sample
    obs = self.actor_obs_normalizer(obs)
    self.update_distribution(obs)
    return self.distribution.sample()  # a ~ N(μ, σ)

def act_inference(self, obs):          # Deploy: mean only
    obs = self.actor_obs_normalizer(obs)
    return self.actor(obs)             # return μ directly
```

**σ (std) is a learnable parameter**:
- Initially `std = 1.0` → "wide" policy → high exploration.
- During training, PPO automatically adjusts std to an optimal level.
- Smaller std → "more confident" policy → less exploration.

**Log-probability** — required for PPO loss:
```python
def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)
    # log π(a|s) = Σ log N(a_i; μ_i, σ_i)   (each joint independent)
```

### 3.4. Critic — State Evaluation

```python
# actor_critic.py lines 153-156
def evaluate(self, obs, **kwargs):
    obs = self.critic_obs_normalizer(obs)
    return self.critic(obs)   # V(s): "how much reward is expected from this state?"
```

**Information Asymmetry — Actor vs Critic:**

| | Actor (deploy) | Critic (training only) |
|--|---------------|----------------------|
| IMU angular velocity | ✓ | ✓ |
| Projected gravity | ✓ | ✓ |
| Joint pos/vel | ✓ | ✓ |
| Commands | ✓ | ✓ |
| **Linear velocity (Ground Truth)** | ✗ | ✓ |

The Critic has "privileged" information (not available on the real robot) → more accurate V(s) estimation → more accurate advantage → better training. Only the Actor is needed for deployment.

### 3.5. Observation Normalization

```python
# actor_critic.py lines 58-72
if actor_obs_normalization:
    self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
```

**Problem:** Observations have very different scales:
- Joint angles: `[-3, 3]` rad
- Angular velocity: `[-10, 10]` rad/s
- Gravity vector: `[-1, 1]`
- Commands: `[-2, 2]` m/s

Without normalization, gradients from larger-scale observations will dominate.

**Solution:** Running normalization — online update of mean/variance:
```
obs_normalized = (obs - running_mean) / (running_std + ε)
```
The normalizer learns from data without needing predefined min/max values.

### 3.6. Default MLP Architecture (G1)

```
Actor:
  input(~50) → Linear(50,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,23)
                                                                                         └─ 23 joint targets

Critic:
  input(~53) → Linear(53,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,1)
                                                                                         └─ scalar V(s)
```

**Why ELU instead of ReLU?**
- ReLU: `max(0, x)` — gradient = 0 for x < 0 (dying ReLU).
- ELU: `x if x>0, α(e^x - 1) if x<0` — gradient never reaches 0.
- Smoother, converges faster for continuous control.

---

## 4. PPO — Problems and Solutions

**File:** `mjlab/rsl_rl/algorithms/ppo.py`

### 4.1. Problems PPO Solves

**Basic Policy Gradient** (REINFORCE, A2C):
```python
# Simple policy update
loss = -log_prob * advantage
optimizer.step()
```

**Problem 1: Excessive update steps can destroy the policy**

```
Good Policy (iteration 1000)
    → Large Update
    → Policy deteriorates significantly
    → Hard to recover because new data is collected by a bad policy
    → Diverge
```

Policy gradient has no explicit "step size" limit — one bad gradient step can destroy everything learned.

**Problem 2: Sample inefficiency with vanilla policy gradient**

Each batch of data can only be used **once** for an update (on-policy). Updating multiple times on the same data → distribution shift → gradients are no longer valid.

### 4.2. TRPO Solution (PPO's Predecessor)

TRPO (Trust Region Policy Optimization) solves this with an **explicit constraint**:
```
maximize:  E[ ratio · A ]
subject to: E[ KL(π_old || π_new) ] ≤ δ
```

Good in theory but **too complex**: requires natural gradient calculation, conjugate gradient solver, line search — hard to implement and slow.

### 4.3. PPO Solution: Simple, Effective Clipping

PPO replaces TRPO's constraint with **clipping in the objective function**:

```python
# ppo.py lines 297-302
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
# ratio = π_new(a|s) / π_old(a|s)

surrogate        = -advantages_batch * ratio
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1-ε, 1+ε)

surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

**Visualizing Clipping:**

```
When Advantage > 0 (GOOD action, want to increase probability):

  -A·ratio
     │
     │  ╲
     │   ╲__________  ← clip: prevents ratio from increasing past 1+ε
     │              ╲
     └──────────────────→ ratio
        1-ε    1   1+ε

When Advantage < 0 (BAD action, want to decrease probability):

  -A·ratio  (Negative A → reversed slope)
     │
     │  __________
     │ ╱           ← clip: prevents ratio from decreasing past 1-ε
     │╱
     └──────────────────→ ratio
        1-ε    1   1+ε
```

`torch.max(surrogate, surrogate_clipped)` = takes the **more pessimistic** value:
- Clipping prevents gradients from pushing the policy outside the "trust region".
- No need for complex constrained optimization solvers.

### 4.4. Why can we update multiple times on the same data?

```
Collect 98,304 transitions using π_old
                    │
                    ▼
    Update π_new: epoch 1, batch 1  → ratio near 1.0, OK
    Update π_new: epoch 1, batch 2  → ratio slightly far from 1.0, clipping starts
    ...
    Update π_new: epoch 5, batch 4  → ratio heavily clipped → gradient shrinks
                                                              → update naturally stops
```

Clipping **automatically controls** the number of useful updates — when the policy becomes too different from π_old, the gradient is clipped → further updates have no effect → safety guaranteed.

### 4.5. Value Function Loss

```python
# ppo.py lines 305-313
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

The Critic learns to predict `V(s)` via supervised learning:
- **Target**: `returns_batch` = GAE returns (R_t).
- **Prediction**: `value_batch` = current V(s).
- **Loss**: MSE = (prediction - target)².

Clipped value loss: Similar to the Actor — limits the Critic from changing too rapidly.

### 4.6. Entropy Bonus — Encouraging Exploration

```python
# ppo.py line 315
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()
```

**Entropy of a Gaussian:** `H(N(μ,σ)) = 0.5 * log(2πeσ²)`

- Large σ → high entropy → diverse policy → more exploration.
- Subtracting entropy from loss = **rewarding** policies with high entropy.
- Small `entropy_coef = 0.01` → mild encouragement, prevents policy from becoming purely random.

### 4.7. Total Loss

```python
L_total = L_surrogate + c₁·L_value - c₂·H(π)
        =    (actor)  +  (critic)  +  (exploration)
```

| Component | Purpose | Coefficient |
|-----------|---------|-------|
| `L_surrogate` | Policy optimization with clipping | 1.0 |
| `c₁·L_value` | Critic training | `value_loss_coef` = 1.0 |
| `-c₂·H(π)` | Maintain entropy, avoid premature convergence | `entropy_coef` = 0.01 |

### 4.8. Pros and Cons of PPO

**Pros:**
- **Simple**: No second-order optimization like TRPO.
- **Stable**: Clipping prevents updates from destroying the policy.
- **Sample reuse**: Safely reuse data over multiple epochs.
- **Parallel-friendly**: Naturally scales with many environments.
- **Robust**: Less sensitive to hyperparameters than many other algorithms.

**Cons:**
- **On-policy**: Each rollout is used once and discarded → less sample efficient than SAC.
- **Sensitive to reward scale**: Large/small rewards affect advantage estimation.
- **Not optimal in sample efficiency**: For the same data volume, SAC often learns better.
- **Hard to tune for continuous action**: std schedule and clip_param have significant impact.

**Comparison with SAC (Soft Actor-Critic):**

| | PPO | SAC |
|--|-----|-----|
| Type | On-policy | Off-policy |
| Sample efficiency | Medium | High |
| Replay buffer | No | Yes |
| Entropy | Heuristic bonus | Explicit in objective |
| Sim-to-Real | Good | Good |
| Multi-env parallel | Excellent | More complex |
| When to choose | Sim with many envs | Sim with few envs or real robot |

---

## 5. GAE — Advantage Estimation

**File:** `mjlab/rsl_rl/storage/rollout_storage.py` lines 127-149

### 5.1. Problem: Bias-Variance Tradeoff

To calculate Advantage `A(s,a) = Q(s,a) - V(s)`, we need to estimate Q(s,a). There are two extremes:

**Method 1 — Monte Carlo (λ=1):**
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... (until end of episode)
A_t = G_t - V(s_t)
```
- **Bias = 0**: No systematic error (unbiased).
- **High Variance**: Depends on the entire sequence of random rewards.
- Requires long episodes → slow.

**Method 2 — 1-step TD (λ=0):**
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t
```
- **Low Variance**: Uses only one step.
- **High Bias**: Depends on the accuracy of V(s) (bootstrapping bias).

**GAE** (λ=0.95): A weighted combination of all n-step estimates:
```
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

### 5.2. Source Code Implementation

```python
# rollout_storage.py lines 127-149
def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):  # Backward from T→0

        next_values = last_values if step == T-1 else self.values[step + 1]

        # 1 if not a terminal state
        next_is_not_terminal = 1.0 - self.dones[step].float()

        # TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]

        # GAE: A_t = δ_t + γλ·A_{t+1}  (backward recursion)
        advantage = delta + next_is_not_terminal * gamma * lam * advantage

        # Return: R_t = A_t + V(s_t)
        self.returns[step] = advantage + self.values[step]

    self.advantages = self.returns - self.values

    if normalize_advantage:
        # Normalize: mean=0, std=1
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

**Why reversed?**

GAE is recursive: `A_t` needs `A_{t+1}`, `A_{t+1}` needs `A_{t+2}`, etc. Iterating backward from T to 0 ensures `A_{t+1}` is available when calculating `A_t`.

**Why normalize advantages?**

Advantages can have very different values across episodes. Normalizing to mean=0, std=1:
- Stabilizes gradients.
- Prevents a few transitions with extreme advantages from dominating the update.

### 5.3. Bootstrapping on Timeout

```python
# ppo.py lines 160-163
if "time_outs" in extras:
    self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1), 1
    )
```

Distinguishing between two types of episode endings:
- **Termination** (falling, collision): Robot truly failed → V(s_terminal) = 0.
- **Truncation** (timeout at 20s): Robot is still functional, just cut off → V(s) ≠ 0.

If truncation isn't handled, the policy will learn that "the state at the end of an episode is always bad" → severe bias.

Solution: Add `γ·V(s)` to the final reward on timeout → the policy sees that "if it continued, more reward would be received".

---

## 6. Rollout Storage — Experience Memory

**File:** `mjlab/rsl_rl/storage/rollout_storage.py`

### 6.1. Buffer Structure

The buffer stores `T × N` transitions before an update (T=24 steps, N=4096 envs):

```python
# rollout_storage.py lines 31-68
self.observations    = TensorDict(...)           # [T, N, obs_dim]
self.actions         = torch.zeros(T, N, act)    # [T, N, 23]
self.rewards         = torch.zeros(T, N, 1)      # [T, N, 1]
self.dones           = torch.zeros(T, N, 1)      # [T, N, 1]
self.values          = torch.zeros(T, N, 1)      # [T, N, 1]   ← V(s_t)
self.actions_log_prob = torch.zeros(T, N, 1)     # [T, N, 1]   ← log π(a|s)
self.mu              = torch.zeros(T, N, act)    # [T, N, 23]  ← action mean
self.sigma           = torch.zeros(T, N, act)    # [T, N, 23]  ← action std
self.returns         = torch.zeros(T, N, 1)      # [T, N, 1]   ← R_t (after GAE)
self.advantages      = torch.zeros(T, N, 1)      # [T, N, 1]   ← A_t (after GAE)
```

### 6.2. Mini-batch Generator — Why shuffle?

```python
# rollout_storage.py lines 160-203
def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    batch_size = N * T  # 4096 * 24 = 98,304

    # Randomly shuffle the entire buffer
    indices = torch.randperm(batch_size)

    for epoch in range(num_epochs):      # 5 epochs
        for i in range(num_mini_batches): # 4 mini-batches
            batch_idx = indices[start:end]   # 24,576 samples
            yield obs[batch_idx], actions[batch_idx], ...
```

**Why shuffle?** Sequential transitions are highly correlated (s_t and s_{t+1} are very similar). Training in temporal order leads to biased gradients. Shuffling breaks temporal correlation → more diverse gradients.

---

## 7. Training Loop

**File:** `mjlab/rsl_rl/runners/on_policy_runner.py` lines 62-176

### 7.1. Main Loop

```python
def learn(self, num_learning_iterations):
    obs = self.env.get_observations()

    for it in range(num_learning_iterations):  # 10,001 iterations

        # ═══ PHASE 1: ROLLOUT ═══════════════════════════════
        with torch.inference_mode():            # No gradient calculation
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

### 7.2. Phase 1: `alg.act()` — Data Collection

```python
# ppo.py lines 129-140
def act(self, obs):
    self.transition.actions          = self.policy.act(obs).detach()
    self.transition.values           = self.policy.evaluate(obs).detach()   # V(s_t)
    self.transition.actions_log_prob = self.policy.get_actions_log_prob(...).detach()  # log π_old(a|s)
    self.transition.action_mean      = self.policy.action_mean.detach()    # μ (needed for KL)
    self.transition.action_sigma     = self.policy.action_std.detach()     # σ (needed for KL)
    self.transition.observations     = obs
    return self.transition.actions
```

`detach()` — **Critical**: Decouples tensors from the computation graph. The rollout phase doesn't need gradients, only numerical values for storage.

### 7.3. Phase 2: `alg.update()` — Policy Update

```python
# ppo.py lines 178-422 (simplified)
def update(self):
    generator = self.storage.mini_batch_generator(4, 5)  # 4 batches, 5 epochs

    for obs_batch, actions_batch, values_batch, advantages_batch, returns_batch,
        old_log_prob_batch, old_mu_batch, old_sigma_batch, ... in generator:

        # ── 1. Forward pass with NEW policy ──────────────────
        self.policy.act(obs_batch)
        new_log_prob  = self.policy.get_actions_log_prob(actions_batch)
        new_value     = self.policy.evaluate(obs_batch)
        entropy       = self.policy.entropy

        # ── 2. KL → Adaptive LR ─────────────────────────────
        # (see section 8)

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

### 7.4. 1-Iteration Statistics Summary

```
1 Iteration (G1 velocity):
├── Rollout:
│   ├── 24 steps × 4096 envs = 98,304 transitions
│   ├── Simulation time: 24 × 0.02s = 0.48s robot time
│   └── GAE advantage calculation for all transitions
│
├── Update:
│   ├── 5 epochs × 4 mini-batches = 20 gradient steps
│   ├── Mini-batch size: 98,304 / 4 = 24,576
│   └── ~20 backward passes
│
└── Total training:
    ├── 10,001 iterations × 98,304 = ~983M transitions
    └── 10,001 × 20 = 200,020 gradient steps
```

---

## 8. Adaptive Learning Rate

**File:** `mjlab/rsl_rl/algorithms/ppo.py` lines 260-294

### 8.1. Problem with fixed learning rate

- LR too large → rapid policy changes, unstable, potential divergence.
- LR too small → slow learning, wasted compute.
- A good initial LR may not be optimal in later stages.

### 8.2. Solution: Adjust based on KL divergence

KL divergence measures the "distance" between two distributions (new vs. old policy):

```python
# ppo.py lines 262-269
kl = torch.sum(
    torch.log(sigma_new / sigma_old + 1e-5)         # log(σ_new/σ_old)
    + (sigma_old**2 + (mu_old - mu_new)**2)          # (σ_old² + (μ_old-μ_new)²)
      / (2 * sigma_new**2)                           # ÷ 2σ_new²
    - 0.5,
    axis=-1,
)
# This is KL(N(μ_old,σ_old) || N(μ_new,σ_new)) — closed-form for Gaussian
```

```python
# ppo.py lines 280-284
if kl_mean > desired_kl * 2.0:          # Policy changed TOO MUCH
    lr = max(1e-5, lr / 1.5)            # → Decrease LR by 1.5x
elif kl_mean < desired_kl / 2.0:        # Policy changed TOO LITTLE
    lr = min(1e-2, lr * 1.5)            # → Increase LR by 1.5x
# Else: KL within [0.005, 0.02] → maintain LR
```

With `desired_kl = 0.01`:
```
KL > 0.02  → lr /= 1.5   (Brake)
KL < 0.005 → lr *= 1.5   (Accelerate)
```

This mechanism operates **parallel to clipping** — clipping limits on a per-sample basis, while adaptive LR adjusts global learning speed.

---

## 9. Multi-GPU Training

**File:** `mjlab/rsl_rl/algorithms/ppo.py` lines 428-469

### 9.1. Architecture

Each GPU runs its own set of environments, collects data independently, and **synchronizes gradients** before the update:

```
GPU 0 (4096 envs): rollout → gradients_0 ─┐
GPU 1 (4096 envs): rollout → gradients_1 ─┼─→ all_reduce (avg) ─→ update params
GPU 2 (4096 envs): rollout → gradients_2 ─┘

Result: Each GPU updates with average gradients across 4096×3 = 12,288 envs.
        Policies on all GPUs remain IDENTICAL after each update.
```

### 9.2. Parameter Broadcast (Once before training)

```python
# ppo.py lines 428-439
def broadcast_parameters(self):
    model_params = [self.policy.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)  # GPU 0 → all
    self.policy.load_state_dict(model_params[0])
```

### 9.3. Gradient Reduction (Each gradient step)

```python
# ppo.py lines 441-469
def reduce_parameters(self):
    grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
    all_grads = torch.cat(grads)                                      # Flatten
    torch.distributed.all_reduce(all_grads, op=ReduceOp.SUM)         # Sum all
    all_grads /= self.gpu_world_size                                   # Average
    # Copy back to param.grad
```

### 9.4. KL Divergence in Multi-GPU

```python
# ppo.py lines 272-275
if self.is_multi_gpu:
    torch.distributed.all_reduce(kl_mean, op=ReduceOp.SUM)
    kl_mean /= self.gpu_world_size
# → Average KL across all GPUs → consistent adaptive LR
```

---

## 10. Advanced Techniques

### 10.1. RND — Random Network Distillation (Exploration bonus)

**Problem:** Sparse reward — the robot doesn't receive rewards until it achieves a difficult task. Policy gradient cannot learn from reward = 0 indefinitely.

**Solution:** Generate an **intrinsic reward** based on state "novelty":

```python
# ppo.py lines 152-158
if self.rnd:
    intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
    self.transition.rewards += intrinsic_rewards
    # Total reward = extrinsic (env) + intrinsic (RND)
```

**Mechanism:**
```
Target network  (frozen random MLP): obs → embedding_target   ← No training
Predictor network (trained MLP):     obs → embedding_predict  ← Trained to match target

Intrinsic reward = ||embedding_predict - embedding_target||²
```

- **Frequently visited** states: predictor has learned → low prediction error → low reward.
- **Novel** states: predictor hasn't learned → high prediction error → high reward → encourages exploration.

### 10.2. Symmetry Augmentation

**Problem:** Locomotion robots are left-right symmetric. Learning both sides effectively requires double the data.

**Solution:** Data augmentation via left-right flipping:

```python
# ppo.py lines 226-244
if self.symmetry["use_data_augmentation"]:
    obs_batch, actions_batch = data_augmentation_func(obs=obs_batch, actions=actions_batch)
    # Flipped obs: joint left ↔ right, IMU sign flip
    # Flipped actions: joint command left ↔ right
    # Batch size doubles: [B] → [2B]
```

Result: Policy learns twice as fast because every left step also teaches it how to take a right step.

### 10.3. Gradient Clipping

```python
# ppo.py line 380
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # max = 1.0
```

If the total gradient norm > 1.0, scale it down:
```
grad = grad × (1.0 / ||grad||)  when ||grad|| > 1.0
```

Prevents "gradient explosion" — often caused by sudden extreme rewards (e.g., −200 penalty for falling) → massive gradients → abrupt parameter changes → network collapse.

---

## 11. Data Flow Summary

### Full Pipeline in 1 Iteration

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
│  Compute GAE (backward from T→0):                           │
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

### Default Hyperparameters (G1 velocity)

| Parameter | Value | Meaning |
|-----------|---------|---------|
| `num_envs` | 4096 | Number of parallel robots |
| `num_steps_per_env` | 24 | Rollout steps per iteration |
| `max_iterations` | 10001 | Total iterations |
| `gamma` | 0.99 | Discount (horizon ~100 steps) |
| `lam` | 0.95 | GAE lambda (bias/variance) |
| `clip_param` | 0.2 | PPO clipping ε |
| `learning_rate` | 1e-3 | Initial LR (adaptive later) |
| `desired_kl` | 0.01 | Target KL for adaptive LR |
| `num_learning_epochs` | 5 | Epochs per update |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_loss_coef` | 1.0 | Value loss coefficient |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `actor_hidden_dims` | (512, 256, 128) | Actor MLP architecture |
| `critic_hidden_dims` | (512, 256, 128) | Critic MLP architecture |
| `init_noise_std` | 1.0 | Initial noise (exploration) |
| `activation` | ELU | Activation function |

### Throughput

```
1 iteration  = 24 × 4,096 = 98,304 transitions
10,001 iters = ~983 million total transitions
Per iter     = 20 gradient steps (5 epochs × 4 batches)
Total        = ~200,000 gradient steps
```
