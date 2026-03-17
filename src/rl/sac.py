"""
Soft Actor-Critic (SAC) implemented from scratch.

Key components:
  - Squashed Gaussian actor with tanh (reparameterization trick)
  - Twin Q-networks (clipped double-Q) for the critic
  - Automatic entropy coefficient tuning (ent_coef="auto")
  - Polyak-averaged target networks
  - Single-environment training loop with callbacks

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019.
"""

import copy
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.replay_buffer import ReplayBuffer
from rl.callbacks import BaseCallback

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Neural network components
# ═══════════════════════════════════════════════════════════════════════════════


class GaussianActor(nn.Module):
    """Squashed Gaussian policy: obs → tanh(N(μ(obs), σ(obs)))."""

    def __init__(self, obs_dim: int, action_dim: int, net_arch: List[int] = (256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with tanh squashing. Returns (action, log_prob)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()                                     # reparameterization
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    """Two independent Q-networks: (obs, action) → (Q1, Q2)."""

    def __init__(self, obs_dim: int, action_dim: int, net_arch: List[int] = (256, 256)):
        super().__init__()
        self.q1 = self._build(obs_dim + action_dim, net_arch)
        self.q2 = self._build(obs_dim + action_dim, net_arch)

    @staticmethod
    def _build(in_dim: int, net_arch: List[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = in_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC algorithm
# ═══════════════════════════════════════════════════════════════════════════════


class SAC:
    """
    Soft Actor-Critic for continuous action spaces.

    Provides the same public API as stable-baselines3's SAC:
      - learn(total_timesteps, callback, log_interval)
      - predict(obs, deterministic)
      - save(path) / SAC.load(path)
      - model.replay_buffer  (with .pos, .rewards, .buffer_size)
    """

    def __init__(
        self,
        policy: str,                       # "MlpPolicy" — kept for API compat
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        net_arch: Optional[List[int]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "auto",
        tensorboard_log: Optional[str] = None,
        **kwargs,
    ):
        # Environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert isinstance(self.action_space, gym.spaces.Box)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef_init = ent_coef
        self.target_update_interval = target_update_interval
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        # Device
        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

        # Seeding
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # State
        self.num_timesteps = 0
        self._n_updates = 0
        self._episode_num = 0

        # Network dimensions
        obs_dim = int(np.prod(self.observation_space.shape))
        action_dim = int(np.prod(self.action_space.shape))
        arch = net_arch or [256, 256]

        # Actor
        self.actor = GaussianActor(obs_dim, action_dim, arch).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Critic (twin Q) + target
        self.critic = TwinQNetwork(obs_dim, action_dim, arch).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Entropy coefficient
        self.target_entropy = -float(action_dim)
        if ent_coef == "auto":
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device=self.device)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=learning_rate)
            self._auto_ent = True
        else:
            self.log_ent_coef = torch.tensor(float(ent_coef), device=self.device).log()
            self.ent_coef_optimizer = None
            self._auto_ent = False

        # Action scaling (actor outputs in [-1, 1]; scale to env bounds)
        self._act_low = torch.as_tensor(self.action_space.low, device=self.device).float()
        self._act_high = torch.as_tensor(self.action_space.high, device=self.device).float()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size, self.observation_space, self.action_space,
            device=self.device, n_envs=1,
        )

        # Logging
        self.ep_info_buffer: deque = deque(maxlen=100)

    # ── Action scaling ────────────────────────────────────────────────────────

    def _to_env_action(self, raw: np.ndarray) -> np.ndarray:
        """[-1, 1] → env action range."""
        lo, hi = self.action_space.low, self.action_space.high
        return lo + (raw + 1.0) * 0.5 * (hi - lo)

    def _to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        """env action range → [-1, 1]."""
        lo, hi = self.action_space.low, self.action_space.high
        return 2.0 * (env_action - lo) / (hi - lo) - 1.0

    # ── Core training loop ────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: Optional[int] = None,
    ) -> "SAC":
        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start()

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()
        steps_since_rollout_start = 0

        if callback is not None:
            callback.on_rollout_start()

        for step in range(total_timesteps):
            # Select action
            if self.num_timesteps < self.learning_starts:
                action = self.action_space.sample()
                buffer_action = self._to_buffer_action(action)
            else:
                buffer_action, action = self._sample_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.num_timesteps += 1
            episode_reward += reward
            episode_length += 1
            steps_since_rollout_start += 1

            # Store transition (buffer stores actions in [-1, 1])
            self.replay_buffer.add(
                np.array([obs]),
                np.array([next_obs]),
                np.array([buffer_action]),
                np.array([reward]),
                np.array([done]),
                [info],
            )

            if callback is not None:
                if not callback.on_step():
                    break

            obs = next_obs

            # Handle episode end
            if done:
                self._episode_num += 1
                self.ep_info_buffer.append({"r": episode_reward, "l": episode_length})
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # End of rollout (train_freq steps collected) → callback + train
            if steps_since_rollout_start >= self.train_freq:
                if callback is not None:
                    callback.on_rollout_end()

                if self.num_timesteps > self.learning_starts:
                    self._train(self.gradient_steps)

                steps_since_rollout_start = 0
                if callback is not None:
                    callback.on_rollout_start()

            # Logging
            if (
                self.verbose >= 1
                and log_interval is not None
                and self._episode_num > 0
                and self._episode_num % log_interval == 0
                and done
            ):
                elapsed = time.time() - start_time
                fps = int(self.num_timesteps / max(elapsed, 1e-8))
                recent = list(self.ep_info_buffer)[-log_interval:]
                mean_r = np.mean([e["r"] for e in recent]) if recent else 0
                mean_l = np.mean([e["l"] for e in recent]) if recent else 0
                ent = self.log_ent_coef.exp().item()
                print(
                    f"  Step {self.num_timesteps:>8d} | "
                    f"Episodes: {self._episode_num} | "
                    f"Mean reward: {mean_r:.2f} | "
                    f"Mean length: {mean_l:.0f} | "
                    f"Ent coef: {ent:.4f} | "
                    f"FPS: {fps}"
                )

        # Final rollout end
        if callback is not None:
            callback.on_rollout_end()
            callback.on_training_end()

        return self

    def _sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (buffer_action [-1,1], env_action [lo,hi])."""
        obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            raw, _ = self.actor.sample(obs_t)
            raw_np = raw.cpu().numpy()[0]
        env_action = self._to_env_action(raw_np)
        env_action = np.clip(env_action, self.action_space.low, self.action_space.high)
        return raw_np, env_action

    def _train(self, gradient_steps: int) -> None:
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            ent_coef = self.log_ent_coef.exp().detach()

            # ── Critic loss ──
            with torch.no_grad():
                next_a, next_logp = self.actor.sample(batch.next_observations)
                tq1, tq2 = self.critic_target(batch.next_observations, next_a)
                target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * (
                    torch.min(tq1, tq2) - ent_coef * next_logp
                )
            q1, q2 = self.critic(batch.observations, batch.actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ── Actor loss ──
            a_pi, logp_pi = self.actor.sample(batch.observations)
            q1_pi, q2_pi = self.critic(batch.observations, a_pi)
            actor_loss = (ent_coef * logp_pi - torch.min(q1_pi, q2_pi)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ── Entropy coefficient ──
            if self._auto_ent:
                ent_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_loss.backward()
                self.ent_coef_optimizer.step()

            # ── Target update ──
            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                with torch.no_grad():
                    for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                        tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self, observation: np.ndarray, deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis]
        obs_t = torch.as_tensor(obs, device=self.device).float()

        with torch.no_grad():
            if deterministic:
                raw = self.actor.deterministic(obs_t)
            else:
                raw, _ = self.actor.sample(obs_t)

        raw_np = raw.cpu().numpy()
        action = self._to_env_action(raw_np)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return (action[0], None) if single else (action, None)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_ent_coef": self.log_ent_coef.data,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "num_timesteps": self.num_timesteps,
                "hparams": {
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "batch_size": self.batch_size,
                    "buffer_size": self.buffer_size,
                    "train_freq": self.train_freq,
                    "gradient_steps": self.gradient_steps,
                    "ent_coef": self.ent_coef_init,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None, device: str = "auto", **kwargs) -> "SAC":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if env is None:
            # Build a lightweight stub env so we can reconstruct the networks
            obs_sp = ckpt["observation_space"]
            act_sp = ckpt["action_space"]
            env = _StubEnv(obs_sp, act_sp)

        hparams = ckpt.get("hparams", {})
        model = cls("MlpPolicy", env, device=device, **hparams, **kwargs)
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
        model.critic_target.load_state_dict(ckpt["critic_target"])
        model.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        model.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if "log_ent_coef" in ckpt:
            model.log_ent_coef.data.copy_(ckpt["log_ent_coef"])
        model.num_timesteps = ckpt.get("num_timesteps", 0)
        return model


class _StubEnv:
    """Minimal env-like object used by SAC.load() when no real env is provided."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, **kw):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}

    def close(self):
        pass
