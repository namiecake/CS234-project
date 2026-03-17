"""
Collect rendered frames from SAC checkpoints for EPIC distance evaluation.

Rolls out trained (and optionally random) policies in the textured humanoid
environment, renders every frame at 224x224, and saves them for later labeling.

Usage:
    # Collect from multiple checkpoints + random policy
    python src/collect_frames.py \
        --checkpoints results/.../best_model.zip results/.../ckpt_1280000.zip \
        --include_random \
        --episodes_per_checkpoint 10 \
        --random_episodes 5 \
        --output_dir results/epic_eval

    # Collect from best model only + random
    python src/collect_frames.py \
        --checkpoints results/.../best_model.zip \
        --include_random \
        --output_dir results/epic_eval
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from rl.sac import SAC

from environments import HumanoidVLMWrapper


def create_env(episode_length: int = 100, render_size: int = 224) -> HumanoidVLMWrapper:
    """Create the textured humanoid env matching the training setup exactly."""
    return HumanoidVLMWrapper(
        reward_model=None,
        render_width=render_size,
        render_height=render_size,
        textured=True,
        episode_length=episode_length,
        batch_rewards=False,
    )


def collect_rollout_frames(
    env: HumanoidVLMWrapper,
    agent=None,
    n_episodes: int = 10,
    episode_length: int = 100,
    source_label: str = "unknown",
) -> tuple:
    """
    Run rollouts and collect every rendered frame.

    Returns:
        frames: list of uint8 arrays (H, W, 3)
        metadata: list of dicts with per-frame source info
    """
    frames = []
    metadata = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(episode_length):
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
            metadata.append({
                "source": source_label,
                "episode": ep,
                "step": step,
            })

            if terminated or truncated:
                break

        print(f"    Episode {ep + 1}/{n_episodes}: {step + 1} steps")

    return frames, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Collect rendered humanoid frames for EPIC evaluation"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="Paths to SAC .zip checkpoint files",
    )
    parser.add_argument(
        "--include_random",
        action="store_true",
        help="Also collect frames from a random policy (for non-goal states)",
    )
    parser.add_argument(
        "--episodes_per_checkpoint",
        type=int,
        default=10,
        help="Number of rollout episodes per checkpoint",
    )
    parser.add_argument(
        "--random_episodes",
        type=int,
        default=5,
        help="Number of episodes with random policy",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="kneeling",
        help="Task name (used for output filenames)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/epic_eval",
        help="Directory to save frames and metadata",
    )
    args = parser.parse_args()

    if not args.checkpoints and not args.include_random:
        parser.error("Provide at least one --checkpoints path or use --include_random")

    os.makedirs(args.output_dir, exist_ok=True)

    all_frames = []
    all_metadata = []
    env = create_env()

    # Collect from each checkpoint
    for ckpt_path in args.checkpoints:
        print(f"\nLoading checkpoint: {ckpt_path}")
        agent = SAC.load(ckpt_path)
        label = os.path.basename(ckpt_path).replace(".zip", "")

        print(f"  Running {args.episodes_per_checkpoint} episodes...")
        frames, meta = collect_rollout_frames(
            env,
            agent=agent,
            n_episodes=args.episodes_per_checkpoint,
            source_label=label,
        )
        all_frames.extend(frames)
        all_metadata.extend(meta)
        print(f"  Collected {len(frames)} frames from {label}")

    # Collect from random policy
    if args.include_random:
        print(f"\nRunning {args.random_episodes} episodes with random policy...")
        frames, meta = collect_rollout_frames(
            env,
            agent=None,
            n_episodes=args.random_episodes,
            source_label="random",
        )
        all_frames.extend(frames)
        all_metadata.extend(meta)
        print(f"  Collected {len(frames)} frames from random policy")

    env.close()

    # Save frames as npz
    frames_array = np.array(all_frames, dtype=np.uint8)
    frames_path = os.path.join(args.output_dir, f"frames_{args.task}.npz")
    np.savez_compressed(frames_path, frames=frames_array)
    print(f"\nSaved {len(frames_array)} frames to {frames_path}")
    print(f"  Shape: {frames_array.shape}, dtype: {frames_array.dtype}")

    # Save metadata
    meta_path = os.path.join(args.output_dir, f"frames_{args.task}_meta.json")
    summary = {
        "task": args.task,
        "total_frames": len(all_metadata),
        "checkpoints": args.checkpoints,
        "include_random": args.include_random,
        "episodes_per_checkpoint": args.episodes_per_checkpoint,
        "random_episodes": args.random_episodes,
        "frames": all_metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Print summary
    sources = {}
    for m in all_metadata:
        src = m["source"]
        sources[src] = sources.get(src, 0) + 1

    print(f"\nSummary:")
    print(f"  Total frames: {len(all_frames)}")
    for src, count in sources.items():
        print(f"  - {src}: {count} frames")

    print(f"\nNext step: label the frames using label_frames.py")
    print(f"  python src/label_frames.py --input {frames_path} --task {args.task} --mode sheets")


if __name__ == "__main__":
    main()
