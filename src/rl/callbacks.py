"""Callback system for RL training loops."""

from typing import Any, Dict


class BaseCallback:
    """
    Base class for training callbacks, matching the stable-baselines3 interface.

    The algorithm calls these hooks at specific points during training:
      - _on_rollout_start(): before collecting env transitions
      - _on_step(): after each env step
      - _on_rollout_end(): after the rollout is complete
      - _on_training_start(): at the very beginning of learn()
      - _on_training_end(): at the very end of learn()

    The callback has access to `self.model` (the algorithm instance)
    and `self.num_timesteps` (total env steps so far).
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.model: Any = None
        self.num_timesteps: int = 0
        self.n_calls: int = 0

    def init_callback(self, model: Any) -> None:
        self.model = model

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def on_step(self) -> bool:
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def on_training_start(self) -> None:
        self._on_training_start()

    def on_training_end(self) -> None:
        self._on_training_end()

    # ── Override these in subclasses ──

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
