"""
Evaluation utilities for VLM-RM experiments.

Implements:
  - EPIC distance computation (Section 4.1, Appendix A)
  - Reward landscape visualization
  - Model scale comparison (Section 4.4)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from vlm_reward import CLIPRewardModel, CLIP_MODELS, compute_epic_distance


def compare_model_scales(
    task_prompt: str = "a humanoid robot kneeling",
    baseline_prompt: str = "a humanoid robot",
    frames: np.ndarray = None,
    human_labels: np.ndarray = None,
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    output_dir: str = "results/plots",
):
    """
    Compare CLIP models of different sizes as reward models (reproduces Figure 4).

    Args:
        frames: Pre-collected frames from humanoid rollouts, shape (N, H, W, 3)
        human_labels: Binary labels (1 = goal state, 0 = not), shape (N,)
        alphas: Regularization strengths to evaluate

    If frames/labels not provided, generates synthetic data for testing.
    """
    print("=" * 60)
    print("MODEL SCALE COMPARISON")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # If no real data, create placeholder
    if frames is None:
        print("  No frames provided — using random frames for demonstration.")
        print("  (Replace with real humanoid rollout frames for meaningful results)")
        frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
        human_labels = np.random.binomial(1, 0.3, 100).astype(float)

    models_to_test = ["RN50", "ViT-L-14", "ViT-H-14", "ViT-bigG-14"]

    # ─── Figure 4a: EPIC distance vs alpha for each model ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epic_results = {}  # {model_name: {alpha: epic_distance}}
    model_params = {
        "RN50": 102e6,
        "ViT-L-14": 428e6,
        "ViT-H-14": 986e6,
        "ViT-bigG-14": 2.5e9,
    }

    for model_name in models_to_test:
        print(f"\n  Testing {model_name}...")
        config = CLIP_MODELS[model_name]
        epic_results[model_name] = {}

        for alpha in alphas:
            try:
                rm = CLIPRewardModel(
                    goal_prompt=task_prompt,
                    baseline_prompt=baseline_prompt,
                    alpha=alpha,
                    device="cuda",
                    **config,
                )

                rewards = rm.reward_from_frames(frames)
                epic = compute_epic_distance(rewards, human_labels)
                epic_results[model_name][alpha] = epic
                print(f"    α={alpha:.2f}: EPIC distance = {epic:.4f}")

            except Exception as e:
                print(f"    α={alpha:.2f}: FAILED ({e})")
                epic_results[model_name][alpha] = None

    # Plot Figure 4a: EPIC vs alpha
    ax = axes[0]
    markers = ["o", "s", "^", "v"]
    for (model_name, results), marker in zip(epic_results.items(), markers):
        valid_alphas = [a for a, e in results.items() if e is not None]
        valid_epics = [results[a] for a in valid_alphas]
        if valid_epics:
            ax.plot(valid_alphas, valid_epics, f"-{marker}", label=model_name, linewidth=2)

    ax.set_xlabel("α (regularization strength)", fontsize=12)
    ax.set_ylabel("EPIC distance", fontsize=12)
    ax.set_title("(a) Goal-baseline regularization\nfor different model sizes", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Figure 4b: EPIC vs model size (at alpha=0)
    ax = axes[1]
    for model_name in models_to_test:
        if 0.0 in epic_results[model_name] and epic_results[model_name][0.0] is not None:
            params = model_params[model_name]
            epic = epic_results[model_name][0.0]
            ax.scatter(np.log10(params), epic, s=100, zorder=5)
            ax.annotate(model_name, (np.log10(params), epic),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("log₁₀(number of parameters)", fontsize=12)
    ax.set_ylabel("EPIC distance", fontsize=12)
    ax.set_title("(b) Reward model performance\nby VLM size (α = 0)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_scale_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved plot to {save_path}")
    plt.close()


def visualize_reward_distributions(
    frames: np.ndarray,
    human_labels: np.ndarray,
    model_name: str = "ViT-bigG-14",
    goal_prompt: str = "a humanoid robot kneeling",
    baseline_prompt: str = "a humanoid robot",
    output_dir: str = "results/plots",
):
    """
    Reproduce Figure 7: Reward distributions for goal vs non-goal states.

    Shows how well CLIP separates goal states from non-goal states.
    """
    config = CLIP_MODELS[model_name]
    rm = CLIPRewardModel(
        goal_prompt=goal_prompt,
        baseline_prompt=baseline_prompt,
        alpha=0.0,
        device="cuda",
        **config,
    )

    rewards = rm.reward_from_frames(frames)

    # Standardize rewards
    rewards_std = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    goal_rewards = rewards_std[human_labels == 1]
    non_goal_rewards = rewards_std[human_labels == 0]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.hist(goal_rewards, bins=30, alpha=0.6, color="steelblue", label="Goal state")
    ax.hist(non_goal_rewards, bins=30, alpha=0.6, color="salmon", label="Not goal state")
    ax.axvline(goal_rewards.mean(), color="steelblue", linestyle="--", linewidth=2,
               label=f"Goal mean ({goal_rewards.mean():.2f})")
    ax.axvline(non_goal_rewards.mean(), color="salmon", linestyle="--", linewidth=2,
               label=f"Non-goal mean ({non_goal_rewards.mean():.2f})")

    ax.set_xlabel("Standardized Reward", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Reward Distribution: {model_name}", fontsize=13)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"reward_distribution_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")
    plt.close()


def collect_humanoid_frames_and_labels(
    model_path: str = None,
    n_episodes: int = 50,
    episode_length: int = 100,
    output_path: str = "results/eval_data.npz",
):
    """
    Collect frames from a trained humanoid agent for EPIC distance evaluation.

    The paper (Appendix B) collects rollouts from trained checkpoints
    and has a human label each frame as goal/non-goal.

    For your project, you can:
    1. Collect frames from random policy + trained policy
    2. Manually label a subset (or label all if <500 frames)
    3. Use the labels for EPIC distance computation
    """
    import gymnasium as gym

    env = gym.make(
        "Humanoid-v4",
        render_mode="rgb_array",
        width=224,
        height=224,
        terminate_when_unhealthy=False,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)

    if model_path:
        from stable_baselines3 import SAC
        agent = SAC.load(model_path)
        print(f"  Loaded trained agent from {model_path}")
    else:
        agent = None
        print("  Using random policy for frame collection")

    all_frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(episode_length):
            if agent:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            all_frames.append(frame)

            if terminated or truncated:
                break

    frames = np.array(all_frames)
    env.close()

    # Save frames for labeling
    np.savez(output_path, frames=frames)
    print(f"  Saved {len(frames)} frames to {output_path}")
    print(f"  Next step: manually label frames as goal/non-goal states")

    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["scale", "distribution", "collect"],
                        default="scale")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained agent (for collect)")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    if args.experiment == "scale":
        compare_model_scales(output_dir=args.output_dir)

    elif args.experiment == "distribution":
        # Would need real frames and labels
        print("Run with --experiment collect first to gather frames.")

    elif args.experiment == "collect":
        collect_humanoid_frames_and_labels(
            model_path=args.model_path,
            output_path="results/eval_data.npz",
        )
