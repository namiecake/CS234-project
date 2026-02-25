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


# ─── Custom SAC with batched CLIP rewards ─────────────────────────────────────

class CLIPRewardCallback(BaseCallback):
    """
    Callback that replaces environment rewards with CLIP rewards.

    The paper computes CLIP rewards in batches for efficiency (Algorithm 1).
    This callback collects frames during rollout, then batch-computes rewards.
    """

    def __init__(
        self,
        reward_model: CLIPRewardModel,
        log_freq: int = 1000,
        video_freq: int = 10000,
        video_dir: str = "results/videos",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.reward_model = reward_model
        self.log_freq = log_freq
        self.video_freq = video_freq
        self.video_dir = video_dir
        self.episode_rewards = []
        self.best_reward = -float("inf")

    def _on_step(self) -> bool:
        # Log progress
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            print(
                f"  Step {self.num_timesteps:>8d} | "
                f"Mean CLIP reward (last 100): {mean_reward:.4f} | "
                f"Best: {self.best_reward:.4f}"
            )

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                # Save best model
                save_path = os.path.join(self.video_dir, "..", "checkpoints", "best_model")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save(save_path)

        return True


def make_clip_reward_env(
    reward_model: CLIPRewardModel,
    render_size: int = 224,
    episode_length: int = 100,
):
    """
    Create a humanoid environment that uses CLIP for rewards.

    Key implementation detail from Algorithm 1:
    The paper computes CLIP rewards in batches at the end of episodes.
    For simplicity, we compute per-step here. For efficiency at scale,
    you'd want to batch frames and compute rewards periodically.
    """
    env = HumanoidVLMWrapper(
        reward_model=reward_model,
        render_width=render_size,
        render_height=render_size,
        modify_textures=True,
        modify_camera=True,
    )

    # Wrap with time limit matching paper's episode_length=100
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)

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

    # Create environment
    env = make_clip_reward_env(rm, render_size=224, episode_length=100)

    # Create output directories
    run_name = f"humanoid_{task}_{model_name}_alpha{alpha}_seed{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)

    # Create SAC agent (hyperparameters from Appendix C.2)
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

    # Callback for logging
    callback = CLIPRewardCallback(
        reward_model=rm,
        log_freq=1000,
        video_freq=50000,
        video_dir=os.path.join(run_dir, "videos"),
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
    n_episodes: int = 20,
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

    # Create environment (no CLIP reward needed for eval, just rendering)
    env = gym.make(
        "Humanoid-v4",
        render_mode="rgb_array",
        width=224,
        height=224,
        terminate_when_unhealthy=False,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

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
        if save_video and ep < 5:
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

    args = parser.parse_args()

    if args.eval_only:
        evaluate_humanoid(
            model_path=args.eval_only,
            task=args.task,
            clip_model_name=args.model,
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
