"""
Phase 3: Humanoid experiments with CLIP reward models.

Trains a MuJoCo humanoid to perform tasks specified by text prompts,
using CLIP cosine similarity as the reward signal.

Tasks attempted in the paper (Table 1):
  ✓ Kneeling (100%)
  ✓ Lotus position (100%)
  ✓ Standing up (100%)
  ✓ Arms raised (100%)
  ✓ Doing splits (100%)
  ✗ Hands on hips (64%)
  ✗ Standing on one leg (0%)
  ✗ Arms crossed (0%)

Usage:
    # Quick test (CPU, small model — won't learn but verifies pipeline)
    python src/train_humanoid.py --task kneeling --model RN50 --total_steps 10000

    # Full run (GPU required, this is the real experiment)
    python src/train_humanoid.py --task kneeling --model ViT-bigG-14 --total_steps 10000000

Notes:
    - The paper uses ViT-bigG-14 which requires ~10GB VRAM
    - Smaller models (RN50, ViT-L-14, ViT-H-14) get 0% success on humanoid tasks
    - Training 10M steps takes ~12-24 hours on a single A100
    - Start with fewer steps (e.g., 1M) to verify learning signal
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

from vlm_reward import CLIPRewardModel, CLIP_MODELS
from environments import HumanoidVLMWrapper


# ─── Task definitions (Table 3 from the paper) ───────────────────────────────

HUMANOID_TASKS = {
    "kneeling": {
        "goal_prompt": "a humanoid robot kneeling",
        "baseline_prompt": "a humanoid robot",
    },
    "lotus": {
        "goal_prompt": "a humanoid robot seated down, meditating in the lotus position",
        "baseline_prompt": "a humanoid robot",
    },
    "standing": {
        "goal_prompt": "a humanoid robot standing up",
        "baseline_prompt": "a humanoid robot",
    },
    "arms_raised": {
        "goal_prompt": "a humanoid robot standing up, with both arms raised",
        "baseline_prompt": "a humanoid robot",
    },
    "splits": {
        "goal_prompt": "a humanoid robot practicing gymnastics, doing the side splits",
        "baseline_prompt": "a humanoid robot",
    },
    "hands_on_hips": {
        "goal_prompt": "a humanoid robot standing up with hands on hips",
        "baseline_prompt": "a humanoid robot",
    },
    "arms_crossed": {
        "goal_prompt": "a humanoid robot standing up, with its arms crossed",
        "baseline_prompt": "a humanoid robot",
    },
    "one_leg": {
        "goal_prompt": "a humanoid robot standing up on one leg",
        "baseline_prompt": "a humanoid robot",
    },
}


# ─── Batched CLIP reward computation (Algorithm 1) ─────────────────────────────


class BatchedCLIPRewardCallback(BaseCallback):
    """
    Batched CLIP reward computation following Algorithm 1
    (Rocamonde et al., ICLR 2024, Appendix C).

    During env rollout the wrapper stores raw rendered frames and emits
    placeholder rewards (0.0).  At the end of each rollout (= one episode
    when train_freq == episode_length) this callback:

      1. Pulls the buffered frames from the environment wrapper.
      2. Runs them through CLIP in a single batched forward pass.
      3. Writes the real rewards back into the SAC replay buffer at the
         correct positions.

    This turns per-step CLIP inference (~5 fps with ViT-bigG-14) into a
    single batched call per episode (~50-100+ effective fps).
    """

    def __init__(
        self,
        reward_model: CLIPRewardModel,
        env: HumanoidVLMWrapper,
        log_freq: int = 1000,
        save_dir: str = "results/checkpoints",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.reward_model = reward_model
        self.env = env
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.episode_rewards: list = []
        self.best_reward = -float("inf")
        self._rollout_buffer_start = 0

    def _on_rollout_start(self) -> None:
        self._rollout_buffer_start = self.model.replay_buffer.pos

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        frames = self.env.get_and_clear_frames()
        if len(frames) == 0:
            return

        # --- single batched CLIP forward pass for the whole episode ---
        rewards = self.reward_model.reward_from_frames(frames)

        # --- retroactively patch the replay buffer ---
        buffer = self.model.replay_buffer
        n = len(rewards)
        positions = (self._rollout_buffer_start + np.arange(n)) % buffer.buffer_size
        buffer.rewards[positions, 0] = rewards

        # --- logging / checkpointing ---
        ep_mean = float(np.mean(rewards))
        self.episode_rewards.append(ep_mean)

        if self.num_timesteps % self.log_freq < n and self.episode_rewards:
            recent = self.episode_rewards[-100:]
            print(
                f"  Step {self.num_timesteps:>8d} | "
                f"Mean CLIP reward (last {len(recent)} ep): {np.mean(recent):.4f} | "
                f"Best: {self.best_reward:.4f}"
            )

        if ep_mean > self.best_reward:
            self.best_reward = ep_mean
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save(os.path.join(self.save_dir, "best_model"))


def make_clip_reward_env(
    render_size: int = 224,
    episode_length: int = 100,
):
    """
    Create a humanoid environment configured for batched CLIP rewards.

    The environment renders and buffers frames but does NOT run CLIP itself;
    BatchedCLIPRewardCallback handles that after the rollout completes.
    """
    env = HumanoidVLMWrapper(
        reward_model=None,
        render_width=render_size,
        render_height=render_size,
        textured=True,
        episode_length=episode_length,
        batch_rewards=True,
    )
    return env


def train_humanoid(
    task: str = "kneeling",
    model_name: str = "ViT-bigG-14",
    alpha: float = 0.0,
    total_steps: int = 10_000_000,
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Train a humanoid agent using CLIP rewards.

    Hyperparameters from paper (Appendix C.2):
    - SAC with τ=0.005, γ=0.95, lr=6e-4
    - Episode length: 100
    - Learning starts: 50,000
    - 100 SAC updates every 100 env steps
    - Checkpoint every 128,000 steps
    """
    print("=" * 60)
    print(f"TRAINING: Humanoid - {task}")
    print(f"  Model: {model_name}")
    print(f"  Alpha: {alpha}")
    print(f"  Steps: {total_steps:,}")
    print(f"  Seed: {seed}")
    print("=" * 60)

    # Get task prompts
    task_config = HUMANOID_TASKS[task]

    # Create CLIP reward model
    clip_config = CLIP_MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rm = CLIPRewardModel(
        goal_prompt=task_config["goal_prompt"],
        baseline_prompt=task_config["baseline_prompt"],
        alpha=alpha,
        device=device,
        **clip_config,
    )

    # Create environment in batched-reward mode: renders + stores frames,
    # returns placeholder reward=0.  The callback handles CLIP inference.
    env = make_clip_reward_env(render_size=224, episode_length=100)

    # Create output directories
    run_name = f"humanoid_{task}_{model_name}_alpha{alpha}_seed{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)

    # Create SAC agent (hyperparameters from Appendix C.2)
    # train_freq=100 matches episode_length so each rollout = one episode.
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=6e-4,
        tau=0.005,
        gamma=0.95,
        batch_size=64,
        learning_starts=50_000,
        train_freq=100,
        gradient_steps=100,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
    )

    # Callback: batched CLIP reward computation (Algorithm 1).
    # Receives a direct reference to the unwrapped env so it can pull
    # the frame buffer after each rollout.
    callback = BatchedCLIPRewardCallback(
        reward_model=rm,
        env=env,
        log_freq=1000,
        save_dir=os.path.join(run_dir, "checkpoints"),
    )

    # Train!
    print(f"\n  Starting training... (this will take a while)")
    start_time = time.time()

    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        log_interval=10,
    )

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed/3600:.1f} hours")

    # Save final model
    final_path = os.path.join(run_dir, "checkpoints", "final_model")
    model.save(final_path)
    print(f"  Saved final model to {final_path}")

    env.close()
    return model, run_dir


def evaluate_humanoid(
    model_path: str,
    task: str = "kneeling",
    clip_model_name: str = "ViT-bigG-14",
    n_episodes: int = 100,
    save_video: bool = True,
    output_dir: str = "results/eval",
):
    """
    Evaluate a trained humanoid agent.
    Renders videos and computes average CLIP reward.
    """
    print("=" * 60)
    print(f"EVALUATING: Humanoid - {task}")
    print("=" * 60)

    # Load model
    model = SAC.load(model_path)

    # Create environment with same textures/camera as training (no CLIP reward needed)
    env = HumanoidVLMWrapper(
        reward_model=None,
        render_width=224,
        render_height=224,
        textured=True,
        episode_length=100,
    )

    # Optionally create CLIP reward model for computing rewards on eval
    task_config = HUMANOID_TASKS[task]
    clip_config = CLIP_MODELS[clip_model_name]
    rm = CLIPRewardModel(
        goal_prompt=task_config["goal_prompt"],
        baseline_prompt=task_config["baseline_prompt"],
        alpha=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **clip_config,
    )

    os.makedirs(output_dir, exist_ok=True)
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_frames = []
        episode_rewards = []

        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            frame = env.render()
            episode_frames.append(frame)

            clip_reward = rm.reward_from_frames(frame)
            episode_rewards.append(clip_reward)

            if terminated or truncated:
                break

        mean_r = np.mean(episode_rewards)
        all_rewards.append(mean_r)
        print(f"  Episode {ep+1}/{n_episodes}: mean CLIP reward = {mean_r:.4f}")

        # Save video for first few episodes
        if save_video and ep < n_episodes:
            import imageio
            video_path = os.path.join(output_dir, f"episode_{ep}.mp4")
            imageio.mimsave(video_path, episode_frames, fps=30)

    print(f"\n  Overall mean CLIP reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid VLM-RM experiments")
    parser.add_argument(
        "--task",
        choices=list(HUMANOID_TASKS.keys()),
        default="kneeling",
        help="Humanoid task to train",
    )
    parser.add_argument(
        "--model",
        choices=list(CLIP_MODELS.keys()),
        default="ViT-bigG-14",
        help="CLIP model (ViT-bigG-14 required for humanoid success)",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="Regularization strength")
    parser.add_argument("--total_steps", type=int, default=10_000_000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="Path to saved model to evaluate (skip training)")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of evaluation episodes")

    args = parser.parse_args()

    if args.eval_only:
        evaluate_humanoid(
            model_path=args.eval_only,
            task=args.task,
            clip_model_name=args.model,
            n_episodes=args.n_episodes,
        )
    else:
        train_humanoid(
            task=args.task,
            model_name=args.model,
            alpha=args.alpha,
            total_steps=args.total_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
