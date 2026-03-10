"""
Interactive labeling tool for humanoid frames.

Supports three modes for different server setups:
  - interactive: matplotlib display with keyboard controls
  - sheets:      generate PNG contact sheets for headless servers
  - enter_labels: type in goal frame indices (after reviewing sheets)

Usage:
    # Generate contact sheets (headless server)
    python src/label_frames.py --input results/epic_eval/frames_kneeling.npz \
        --task kneeling --mode sheets

    # Enter labels after reviewing sheets
    python src/label_frames.py --input results/epic_eval/frames_kneeling.npz \
        --task kneeling --mode enter_labels

    # Interactive labeling (requires display)
    python src/label_frames.py --input results/epic_eval/frames_kneeling.npz \
        --task kneeling --mode interactive

    # Resume from saved progress
    python src/label_frames.py --input results/epic_eval/frames_kneeling.npz \
        --task kneeling --mode interactive --resume
"""

import argparse
import os
import sys

import numpy as np

LABELING_CRITERIA = {
    "kneeling": (
        "Agent must be kneeling with both knees touching the floor.\n"
        "Agent must not be losing balance nor kneeling in the air."
    ),
    "splits": (
        "Agent on the floor doing the side splits.\n"
        "Legs are stretched on the floor."
    ),
    "lotus": (
        "Agent seated down in the lotus position.\n"
        "Both knees are on the floor and facing outwards, "
        "while feet must be facing inwards."
    ),
    "standing": "Agent standing up without falling.",
    "arms_raised": "Agent standing up with both arms raised.",
    "hands_on_hips": (
        "Agent standing up with both hands on the base of the hips.\n"
        "Hands must not be on the chest."
    ),
    "arms_crossed": (
        "Agent standing up with its arms crossed.\n"
        "If the agent has its hands just touching but not crossing, "
        "it is not considered valid."
    ),
    "one_leg": (
        "Agent standing up touching the floor only with one leg and "
        "without losing balance.\n"
        "Agent must not be touching the floor with both feet."
    ),
}


def load_frames(path: str) -> np.ndarray:
    """Load frames from npz file."""
    data = np.load(path)
    return data["frames"]


def load_existing_labels(path: str):
    """Load previously saved labels, or return None."""
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["labels"], data["labeled_mask"]


def save_labels(labels: np.ndarray, labeled_mask: np.ndarray, path: str):
    """Save labels to npz."""
    np.savez(path, labels=labels, labeled_mask=labeled_mask)


def print_progress(labels: np.ndarray, labeled_mask: np.ndarray):
    """Print labeling summary stats."""
    n_labeled = labeled_mask.sum()
    n_total = len(labels)
    n_goal = (labels[labeled_mask] == 1).sum()
    n_non_goal = (labels[labeled_mask] == 0).sum()
    pct_goal = 100 * n_goal / max(n_labeled, 1)
    print(f"  Labeled: {n_labeled}/{n_total} | "
          f"Goal: {n_goal} | Non-goal: {n_non_goal} | "
          f"Goal%: {pct_goal:.1f}%")


# ─── Mode: Contact Sheets ──────────────────────────────────────────────────────


def generate_contact_sheets(
    frames: np.ndarray,
    output_dir: str,
    task: str,
    grid_size: int = 10,
):
    """
    Generate PNG contact sheets with frame indices overlaid.

    Creates grid_size x grid_size grids of frames, each cell labeled
    with its global frame index for later reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = len(frames)
    per_sheet = grid_size * grid_size
    n_sheets = (n_frames + per_sheet - 1) // per_sheet

    sheet_dir = os.path.join(output_dir, f"contact_sheets_{task}")
    os.makedirs(sheet_dir, exist_ok=True)

    print(f"Generating {n_sheets} contact sheets ({grid_size}x{grid_size} = {per_sheet} frames each)...")

    for sheet_idx in range(n_sheets):
        start = sheet_idx * per_sheet
        end = min(start + per_sheet, n_frames)
        batch = frames[start:end]

        fig, axes = plt.subplots(
            grid_size, grid_size,
            figsize=(grid_size * 2, grid_size * 2),
            dpi=100,
        )
        fig.suptitle(
            f"Sheet {sheet_idx} | Frames {start}–{end - 1}",
            fontsize=14, fontweight="bold",
        )

        for i in range(per_sheet):
            row, col = divmod(i, grid_size)
            ax = axes[row][col]
            ax.axis("off")

            if i < len(batch):
                ax.imshow(batch[i])
                ax.set_title(str(start + i), fontsize=7, pad=1)
            else:
                ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(sheet_dir, f"sheet_{sheet_idx:03d}.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")

    print(f"\nAll sheets saved to {sheet_dir}/")
    print(f"Review the sheets, then run:")
    print(f"  python src/label_frames.py --input <frames.npz> --task {task} --mode enter_labels")


# ─── Mode: Enter Labels ────────────────────────────────────────────────────────


def parse_index_ranges(text: str) -> set:
    """
    Parse comma-separated indices and ranges.

    Examples:
        "100-150, 205, 300-350" -> {100, 101, ..., 150, 205, 300, ..., 350}
        "0-9, 50" -> {0, 1, ..., 9, 50}
    """
    indices = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo, hi = int(bounds[0].strip()), int(bounds[1].strip())
                indices.update(range(lo, hi + 1))
            except ValueError:
                print(f"  Warning: could not parse range '{part}', skipping")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                print(f"  Warning: could not parse index '{part}', skipping")
    return indices


def enter_labels_mode(
    frames: np.ndarray,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
    save_path: str,
    task: str,
):
    """
    Prompt user to enter goal frame indices.
    All unlisted frames default to non-goal (0).
    """
    n = len(frames)
    criteria = LABELING_CRITERIA.get(task, "No specific criteria defined.")
    print(f"\nLabeling criteria for '{task}':")
    print(f"  {criteria}")
    print(f"\nTotal frames: {n}")
    print(f"Enter the indices of GOAL frames (all others will be non-goal).")
    print(f"Use comma-separated values and ranges, e.g.: 100-150, 205, 300-350")
    print(f"Enter 'done' when finished, or 'q' to quit without saving.\n")

    all_goal_indices = set()

    while True:
        try:
            line = input("Goal indices> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if line.lower() == "q":
            print("Quit without saving.")
            return
        if line.lower() == "done":
            break

        new_indices = parse_index_ranges(line)
        valid = {i for i in new_indices if 0 <= i < n}
        invalid_count = len(new_indices) - len(valid)
        if invalid_count > 0:
            print(f"  Warning: {invalid_count} indices out of range [0, {n - 1}], ignored")

        all_goal_indices.update(valid)
        print(f"  Added {len(valid)} indices. Total goal indices so far: {len(all_goal_indices)}")

    # Apply labels: everything labeled, goal indices get 1, rest get 0
    labels[:] = 0
    labeled_mask[:] = True
    for idx in all_goal_indices:
        labels[idx] = 1

    save_labels(labels, labeled_mask, save_path)
    print(f"\nSaved labels to {save_path}")
    print_progress(labels, labeled_mask)


# ─── Mode: Interactive ──────────────────────────────────────────────────────────


def interactive_mode(
    frames: np.ndarray,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
    save_path: str,
    task: str,
    resume: bool = False,
):
    """
    Display frames one at a time with matplotlib.
    Keyboard: y=goal, n=non-goal, b=back, s=skip, q=save&quit.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    n = len(frames)
    criteria = LABELING_CRITERIA.get(task, "No specific criteria defined.")

    # Find starting index
    if resume and labeled_mask.any():
        start_idx = int(np.where(~labeled_mask)[0][0]) if not labeled_mask.all() else n
        print(f"Resuming from frame {start_idx}")
    else:
        start_idx = 0

    current = [start_idx]
    done = [False]
    auto_save_interval = 50
    labels_since_save = [0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update_display():
        idx = current[0]
        if idx >= n:
            ax.clear()
            ax.text(0.5, 0.5, "All frames reviewed!\nPress 'q' to save & quit.",
                    ha="center", va="center", fontsize=16, transform=ax.transAxes)
            fig.canvas.draw_idle()
            return

        ax.clear()
        ax.imshow(frames[idx])
        ax.axis("off")

        status = ""
        if labeled_mask[idx]:
            status = " [GOAL]" if labels[idx] == 1 else " [NON-GOAL]"

        n_labeled = int(labeled_mask.sum())
        n_goal = int((labels[labeled_mask] == 1).sum())
        n_non = n_labeled - n_goal

        ax.set_title(
            f"Frame {idx}/{n - 1}{status}\n"
            f"Labeled: {n_labeled}/{n} | Goal: {n_goal} | Non-goal: {n_non}\n"
            f"[y]goal  [n]non-goal  [b]back  [s]skip  [q]save&quit",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        idx = current[0]

        if event.key == "q":
            save_labels(labels, labeled_mask, save_path)
            print(f"\nSaved labels to {save_path}")
            print_progress(labels, labeled_mask)
            done[0] = True
            plt.close(fig)
            return

        if event.key == "y" and idx < n:
            labels[idx] = 1
            labeled_mask[idx] = True
            labels_since_save[0] += 1
            current[0] = min(idx + 1, n)

        elif event.key == "n" and idx < n:
            labels[idx] = 0
            labeled_mask[idx] = True
            labels_since_save[0] += 1
            current[0] = min(idx + 1, n)

        elif event.key == "b":
            current[0] = max(idx - 1, 0)

        elif event.key == "s" and idx < n:
            current[0] = min(idx + 1, n)

        # Auto-save
        if labels_since_save[0] >= auto_save_interval:
            save_labels(labels, labeled_mask, save_path)
            labels_since_save[0] = 0
            print(f"  Auto-saved at frame {current[0]}")

        update_display()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\nLabeling criteria for '{task}':")
    print(f"  {criteria}")
    print(f"\nControls: [y] goal | [n] non-goal | [b] back | [s] skip | [q] save & quit")
    print(f"Starting at frame {start_idx}...\n")

    update_display()
    plt.show()

    if not done[0]:
        save_labels(labels, labeled_mask, save_path)
        print(f"\nWindow closed. Saved labels to {save_path}")
        print_progress(labels, labeled_mask)


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Label humanoid frames for EPIC evaluation")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to frames .npz file (from collect_frames.py)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="kneeling",
        help="Task name (determines labeling criteria displayed)",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "sheets", "enter_labels"],
        default="sheets",
        help="Labeling mode: interactive (matplotlib), sheets (contact PNGs), enter_labels (text input)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from previously saved labels",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to same directory as input file)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Contact sheet grid size (sheets mode only)",
    )
    args = parser.parse_args()

    # Load frames
    print(f"Loading frames from {args.input}...")
    frames = load_frames(args.input)
    n = len(frames)
    print(f"  Loaded {n} frames, shape: {frames.shape}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    labels_path = os.path.join(args.output_dir, f"labels_{args.task}.npz")

    # Initialize or load labels
    if args.resume:
        existing_labels, existing_mask = load_existing_labels(labels_path)
        if existing_labels is not None:
            labels = existing_labels
            labeled_mask = existing_mask
            print(f"  Resumed from {labels_path}")
            print_progress(labels, labeled_mask)
        else:
            print(f"  No existing labels found at {labels_path}, starting fresh")
            labels = np.zeros(n, dtype=np.int32)
            labeled_mask = np.zeros(n, dtype=bool)
    else:
        labels = np.zeros(n, dtype=np.int32)
        labeled_mask = np.zeros(n, dtype=bool)

    # Dispatch to mode
    if args.mode == "sheets":
        generate_contact_sheets(
            frames, args.output_dir, args.task, grid_size=args.grid_size,
        )

    elif args.mode == "enter_labels":
        enter_labels_mode(frames, labels, labeled_mask, labels_path, args.task)

    elif args.mode == "interactive":
        interactive_mode(
            frames, labels, labeled_mask, labels_path, args.task, resume=args.resume,
        )


if __name__ == "__main__":
    main()
