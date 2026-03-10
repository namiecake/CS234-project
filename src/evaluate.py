"""
Evaluation utilities for VLM-RM experiments.

Implements:
  - EPIC distance computation (Section 4.1, Appendix A)
  - Reward landscape visualization
  - Model scale comparison (Section 4.4)

Usage:
    # Compute EPIC distances and produce Figure 4a/4b
    python src/evaluate.py --experiment scale \
        --frames_path results/epic_eval/frames_kneeling.npz \
        --labels_path results/epic_eval/labels_kneeling.npz

    # Reward distribution histograms (Figure 7)
    python src/evaluate.py --experiment distribution \
        --frames_path results/epic_eval/frames_kneeling.npz \
        --labels_path results/epic_eval/labels_kneeling.npz
"""

import argparse
import json
import os
import gc

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from vlm_reward import CLIPRewardModel, CLIP_MODELS, compute_epic_distance


def _encode_frames_batched(
    rm: CLIPRewardModel,
    frames: np.ndarray,
    batch_size: int = 64,
) -> torch.Tensor:
    """Preprocess and encode all frames in batches to avoid OOM."""
    all_embeds = []
    n = len(frames)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = frames[start:end]
        images = rm.preprocess_frames(batch)
        embeds = rm.encode_images(images)
        all_embeds.append(embeds)
        if start % (batch_size * 5) == 0:
            print(f"    Encoded {end}/{n} frames")
    return torch.cat(all_embeds, dim=0)


def compare_model_scales(
    task_prompt: str = "a humanoid robot kneeling",
    baseline_prompt: str = "a humanoid robot",
    frames: np.ndarray = None,
    human_labels: np.ndarray = None,
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    output_dir: str = "results/plots",
    device: str = "cuda",
    batch_size: int = 64,
):
    """
    Compare CLIP models of different sizes as reward models (reproduces Figure 4).

    Optimized: loads each model once, encodes all frames once per model,
    then varies alpha on cached embeddings (4 forward passes instead of 20).
    """
    print("=" * 60)
    print("MODEL SCALE COMPARISON")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    if frames is None:
        print("  ERROR: No frames provided.")
        print("  Run collect_frames.py first, then pass --frames_path and --labels_path.")
        return

    print(f"  Frames: {frames.shape[0]}, Labels: {int(human_labels.sum())} goal / "
          f"{int((human_labels == 0).sum())} non-goal")

    models_to_test = ["RN50", "ViT-L-14", "ViT-H-14", "ViT-bigG-14"]
    model_params = {
        "RN50": 102e6,
        "ViT-L-14": 428e6,
        "ViT-H-14": 986e6,
        "ViT-bigG-14": 2.5e9,
    }

    epic_results = {}  # {model_name: {alpha_str: epic_distance}}

    for model_name in models_to_test:
        print(f"\n  Loading {model_name}...")
        config = CLIP_MODELS[model_name]
        epic_results[model_name] = {}

        try:
            rm = CLIPRewardModel(
                goal_prompt=task_prompt,
                baseline_prompt=baseline_prompt,
                alpha=0.0,
                device=device,
                **config,
            )

            # Encode all frames once for this model
            print(f"  Encoding {len(frames)} frames...")
            image_embeds = _encode_frames_batched(rm, frames, batch_size=batch_size)

            # Evaluate each alpha using cached embeddings
            for alpha in alphas:
                rm.alpha = alpha
                rewards = rm.compute_reward(image_embeds).cpu().numpy()
                epic = compute_epic_distance(rewards, human_labels)
                epic_results[model_name][str(alpha)] = epic
                print(f"    α={alpha:.2f}: EPIC = {epic:.4f}")

            # Free GPU memory before loading next model
            del rm, image_embeds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    FAILED: {e}")
            for alpha in alphas:
                epic_results[model_name][str(alpha)] = None

    # Save results as JSON
    results_path = os.path.join(output_dir, "epic_results.json")
    with open(results_path, "w") as f:
        json.dump(epic_results, f, indent=2)
    print(f"\n  Saved EPIC results to {results_path}")

    # ─── Plot Figure 4a + 4b ───────────────────────────────────────────
    _plot_figure4(epic_results, model_params, alphas, output_dir)


def _plot_figure4(
    epic_results: Dict,
    model_params: Dict,
    alphas: List[float],
    output_dir: str,
):
    """Generate the Figure 4a/4b plot from precomputed EPIC results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = ["o", "s", "^", "v"]
    models_to_test = list(epic_results.keys())

    # Figure 4a: EPIC vs alpha
    ax = axes[0]
    for (model_name, results), marker in zip(epic_results.items(), markers):
        valid_alphas = [a for a in alphas if results.get(str(a)) is not None]
        valid_epics = [results[str(a)] for a in valid_alphas]
        if valid_epics:
            ax.plot(valid_alphas, valid_epics, f"-{marker}",
                    label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("α (regularization strength)", fontsize=12)
    ax.set_ylabel("EPIC distance", fontsize=12)
    ax.set_title("(a) Goal-baseline regularization\nfor different model sizes", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Figure 4b: EPIC vs model size (at alpha=0)
    ax = axes[1]
    for model_name in models_to_test:
        epic_val = epic_results[model_name].get("0.0")
        if epic_val is not None and model_name in model_params:
            params = model_params[model_name]
            ax.scatter(np.log10(params), epic_val, s=100, zorder=5)
            ax.annotate(model_name, (np.log10(params), epic_val),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("log\u2081\u2080(number of parameters)", fontsize=12)
    ax.set_ylabel("EPIC distance", fontsize=12)
    ax.set_title("(b) Reward model performance\nby VLM size (\u03b1 = 0)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_scale_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved plot to {save_path}")
    plt.close()


def plot_from_json(
    results_path: str = "results/plots/epic_results.json",
    output_dir: str = "results/plots",
):
    """Re-plot Figure 4 from saved JSON results (no GPU needed)."""
    with open(results_path) as f:
        epic_results = json.load(f)

    model_params = {
        "RN50": 102e6,
        "ViT-L-14": 428e6,
        "ViT-H-14": 986e6,
        "ViT-bigG-14": 2.5e9,
    }
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    _plot_figure4(epic_results, model_params, alphas, output_dir)


def visualize_reward_distributions(
    frames: np.ndarray,
    human_labels: np.ndarray,
    model_name: str = "ViT-bigG-14",
    goal_prompt: str = "a humanoid robot kneeling",
    baseline_prompt: str = "a humanoid robot",
    output_dir: str = "results/plots",
    device: str = "cuda",
    batch_size: int = 64,
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
        device=device,
        **config,
    )

    image_embeds = _encode_frames_batched(rm, frames, batch_size=batch_size)
    rewards = rm.compute_reward(image_embeds).cpu().numpy()

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


def _load_data(frames_path: str, labels_path: str):
    """Load frames and labels from npz files."""
    print(f"  Loading frames from {frames_path}")
    frames = np.load(frames_path)["frames"]
    print(f"    Shape: {frames.shape}, dtype: {frames.dtype}")

    print(f"  Loading labels from {labels_path}")
    label_data = np.load(labels_path)
    labels = label_data["labels"].astype(float)
    labeled_mask = label_data["labeled_mask"]
    n_labeled = labeled_mask.sum()
    print(f"    Labeled: {n_labeled}/{len(labels)}, "
          f"Goal: {int(labels[labeled_mask].sum())}, "
          f"Non-goal: {int((labels[labeled_mask] == 0).sum())}")

    if not labeled_mask.all():
        print(f"    WARNING: {(~labeled_mask).sum()} unlabeled frames — using only labeled ones")
        frames = frames[labeled_mask]
        labels = labels[labeled_mask]

    return frames, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-RM evaluation (EPIC + plots)")
    parser.add_argument(
        "--experiment",
        choices=["scale", "distribution", "replot"],
        default="scale",
        help="scale: Figure 4a/4b, distribution: Figure 7, replot: re-plot from JSON",
    )
    parser.add_argument("--frames_path", type=str, default=None,
                        help="Path to frames .npz (from collect_frames.py)")
    parser.add_argument("--labels_path", type=str, default=None,
                        help="Path to labels .npz (from label_frames.py)")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Path to epic_results.json (for replot)")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for CLIP image encoding")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.experiment == "scale":
        if not args.frames_path or not args.labels_path:
            parser.error("--frames_path and --labels_path required for scale experiment")
        frames, labels = _load_data(args.frames_path, args.labels_path)
        compare_model_scales(
            frames=frames,
            human_labels=labels,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    elif args.experiment == "distribution":
        if not args.frames_path or not args.labels_path:
            parser.error("--frames_path and --labels_path required for distribution experiment")
        frames, labels = _load_data(args.frames_path, args.labels_path)
        visualize_reward_distributions(
            frames=frames,
            human_labels=labels,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    elif args.experiment == "replot":
        path = args.results_json or os.path.join(args.output_dir, "epic_results.json")
        plot_from_json(results_path=path, output_dir=args.output_dir)
