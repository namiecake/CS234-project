"""
Deep Q-Network (DQN) implemented from scratch.

Key components:
  - Q-network: MLP mapping observations to Q-values for each discrete action
  - Target network: slow-moving copy for stable TD targets (Polyak averaging)
  - Epsilon-greedy exploration with linear annealing
  - Experience replay with uniform sampling
  - Huber (smooth L1) loss for robustness to outliers
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


class QNetwork(nn.Module):
    """MLP Q-network: obs → Q(s, a) for every discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: List[int] = (64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQN:
    """
    Deep Q-Network for discrete action spaces.

    Hyperparameters match the defaults used in train_classic.py and
    stable-baselines3's DQN where applicable.
    """

    def __init__(
        self,
        policy: str,                       # "MlpPolicy" — kept for API compat
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
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
        assert isinstance(self.action_space, gym.spaces.Discrete)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
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
        self._total_timesteps = 0
        self._n_updates = 0
        self._episode_num = 0

        # Networks
        obs_dim = int(np.prod(self.observation_space.shape))
        n_actions = int(self.action_space.n)
        arch = net_arch or [64, 64]

        self.q_net = QNetwork(obs_dim, n_actions, arch).to(self.device)
        self.q_net_target = copy.deepcopy(self.q_net).to(self.device)
        self.q_net_target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size, self.observation_space, self.action_space, device=self.device, n_envs=1,
        )

        # Logging
        self.ep_info_buffer: deque = deque(maxlen=100)

    # ── Exploration schedule ──────────────────────────────────────────────────

    @property
    def exploration_rate(self) -> float:
        progress = self.num_timesteps / max(self._total_timesteps, 1)
        frac = min(1.0, progress / max(self.exploration_fraction, 1e-8))
        return self.exploration_initial_eps + frac * (
            self.exploration_final_eps - self.exploration_initial_eps
        )

    # ── Core training loop ────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: Optional[int] = None,
    ) -> "DQN":
        self._total_timesteps = total_timesteps

        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start()

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()

        for step in range(total_timesteps):
            if callback is not None:
                callback.on_rollout_start()

            # Select action
            if self.num_timesteps < self.learning_starts:
                action = self.action_space.sample()
            else:
                action = self._select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.num_timesteps += 1
            episode_reward += reward
            episode_length += 1

            # Store transition
            self.replay_buffer.add(
                np.array([obs]),
                np.array([next_obs]),
                np.array([action]),
                np.array([reward]),
                np.array([done]),
                [info],
            )

            if callback is not None:
                if not callback.on_step():
                    break
                callback.on_rollout_end()

            obs = next_obs

            # Handle episode end
            if done:
                self._episode_num += 1
                self.ep_info_buffer.append({"r": episode_reward, "l": episode_length})
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Train
            if (
                self.num_timesteps > self.learning_starts
                and self.num_timesteps % self.train_freq == 0
            ):
                self._train(self.gradient_steps)

            # Logging
            if (
                self.verbose >= 1
                and log_interval is not None
                and self._episode_num > 0
                and self._episode_num % log_interval == 0
                and done  # only log at episode boundaries
            ):
                elapsed = time.time() - start_time
                fps = int(self.num_timesteps / max(elapsed, 1e-8))
                recent = list(self.ep_info_buffer)[-log_interval:]
                mean_r = np.mean([e["r"] for e in recent]) if recent else 0
                mean_l = np.mean([e["l"] for e in recent]) if recent else 0
                print(
                    f"  Step {self.num_timesteps:>8d} | "
                    f"Episodes: {self._episode_num} | "
                    f"Mean reward: {mean_r:.2f} | "
                    f"Mean length: {mean_l:.0f} | "
                    f"Eps: {self.exploration_rate:.3f} | "
                    f"FPS: {fps}"
                )

        if callback is not None:
            callback.on_training_end()
        return self

    def _select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()

    def _train(self, gradient_steps: int) -> None:
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_q = self.q_net_target(batch.next_observations).max(dim=1, keepdim=True).values
                target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

            current_q = self.q_net(batch.observations).gather(1, batch.actions.long())
            loss = F.smooth_l1_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                self._polyak_update()

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self, observation: np.ndarray, deterministic: bool = True,
    ) -> Tuple[int, None]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis]
        obs_t = torch.as_tensor(obs, device=self.device).float()
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        action = q_vals.argmax(dim=1).cpu().numpy()
        if observation.ndim == 1 or np.isscalar(observation):
            return int(action[0]), None
        return action, None

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "q_net_target": self.q_net_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "num_timesteps": self.num_timesteps,
                "hparams": {
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "batch_size": self.batch_size,
                    "buffer_size": self.buffer_size,
                    "exploration_final_eps": self.exploration_final_eps,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None, device: str = "auto", **kwargs) -> "DQN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if env is None:
            raise ValueError("Must provide env when loading DQN")
        model = cls("MlpPolicy", env, device=device, **ckpt.get("hparams", {}), **kwargs)
        model.q_net.load_state_dict(ckpt["q_net"])
        model.q_net_target.load_state_dict(ckpt["q_net_target"])
        model.optimizer.load_state_dict(ckpt["optimizer"])
        model.num_timesteps = ckpt.get("num_timesteps", 0)
        return model
