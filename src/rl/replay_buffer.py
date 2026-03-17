"""Replay buffer for off-policy RL algorithms."""

import pickle
from typing import Any, Dict, List, NamedTuple, Union

import numpy as np
import torch
from gymnasium import spaces


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """
    Experience replay buffer that stores transitions for off-policy learning.

    Stores (obs, action, reward, next_obs, done) tuples in a circular buffer.
    Compatible with both single-env and vec-env usage. The `rewards` and
    `observations` arrays are directly accessible for external modification
    (e.g., by the BatchedCLIPRewardCallback).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs

        if isinstance(device, str):
            device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device == "auto"
                else torch.device(device)
            )
        self.device = device

        self.pos = 0
        self.full = False

        obs_shape = observation_space.shape or ()
        action_shape = (
            (1,) if isinstance(action_space, spaces.Discrete) else (action_space.shape or ())
        )

        self.observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.timeouts = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        action = np.asarray(action, dtype=np.float32)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape(self.n_envs, 1)
        else:
            action = action.reshape(self.n_envs, -1)

        self.observations[self.pos] = np.asarray(obs).reshape(
            self.n_envs, *self.observation_space.shape
        )
        self.next_observations[self.pos] = np.asarray(next_obs).reshape(
            self.n_envs, *self.observation_space.shape
        )
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32).flatten()[: self.n_envs]
        self.dones[self.pos] = np.asarray(done, dtype=np.float32).flatten()[: self.n_envs]

        for i in range(min(self.n_envs, len(infos))):
            self.timeouts[self.pos, i] = float(infos[i].get("TimeLimit.truncated", False))

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)

        dones = self.dones[batch_inds, env_inds].copy()
        dones = dones * (1.0 - self.timeouts[batch_inds, env_inds])

        return ReplayBufferSamples(
            observations=torch.as_tensor(
                self.observations[batch_inds, env_inds], device=self.device
            ).float(),
            actions=torch.as_tensor(
                self.actions[batch_inds, env_inds], device=self.device
            ).float(),
            next_observations=torch.as_tensor(
                self.next_observations[batch_inds, env_inds], device=self.device
            ).float(),
            dones=torch.as_tensor(dones, device=self.device).float().unsqueeze(1),
            rewards=torch.as_tensor(
                self.rewards[batch_inds, env_inds], device=self.device
            ).float().unsqueeze(1),
        )

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "observations": self.observations,
                    "next_observations": self.next_observations,
                    "actions": self.actions,
                    "rewards": self.rewards,
                    "dones": self.dones,
                    "timeouts": self.timeouts,
                    "pos": self.pos,
                    "full": self.full,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.observations = data["observations"]
        self.next_observations = data["next_observations"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.timeouts = data["timeouts"]
        self.pos = data["pos"]
        self.full = data["full"]
