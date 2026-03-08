"""
CLIP-based reward model for RL, following Rocamonde et al. (ICLR 2024).

Implements:
  - Basic CLIP cosine similarity reward (Equation 2)
  - Goal-Baseline Regularization (Definition 1)
  - EPIC distance evaluation (Appendix A)
"""

import torch
import torch.nn.functional as F
import numpy as np
import open_clip
from PIL import Image
from typing import Optional


class CLIPRewardModel:
    """CLIP-based reward model for vision-based RL tasks."""

    def __init__(
        self,
        model_name: str = "ViT-bigG-14",
        pretrained: str = "laion2b_s39b_b160k",
        goal_prompt: str = "a humanoid robot kneeling",
        baseline_prompt: Optional[str] = "a humanoid robot",
        alpha: float = 0.0,
        device: str = "cuda",
    ):
        """
        Args:
            model_name: OpenCLIP model name (e.g., "ViT-bigG-14", "ViT-H-14", "ViT-L-14", "RN50")
            pretrained: Pretrained weights identifier
            goal_prompt: Text description of desired task/state
            baseline_prompt: Text description of neutral/default state (for regularization)
            alpha: Regularization strength (0 = no regularization, 1 = full projection)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.alpha = alpha

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).half().eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Precompute text embeddings (only need to do this once)
        with torch.no_grad():
            goal_tokens = self.tokenizer([goal_prompt]).to(device)
            self.goal_embed = self.model.encode_text(goal_tokens)
            self.goal_embed = F.normalize(self.goal_embed, dim=-1)  # g in the paper

            if baseline_prompt is not None:
                baseline_tokens = self.tokenizer([baseline_prompt]).to(device)
                self.baseline_embed = self.model.encode_text(baseline_tokens)
                self.baseline_embed = F.normalize(self.baseline_embed, dim=-1)  # b in the paper
            else:
                self.baseline_embed = None

        print(f"CLIPRewardModel initialized: {model_name} ({pretrained})")
        print(f"  Goal prompt: '{goal_prompt}'")
        print(f"  Baseline prompt: '{baseline_prompt}'")
        print(f"  Alpha: {alpha}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images using CLIP's image encoder.

        Args:
            images: Tensor of shape (B, C, H, W), already preprocessed for CLIP

        Returns:
            Normalized image embeddings of shape (B, D)
        """
        with torch.no_grad():
            image_embeds = self.model.encode_image(images.to(self.device).half())
            image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Convert raw RGB frames (from MuJoCo renderer) to CLIP input format.

        Args:
            frames: numpy array of shape (B, H, W, 3) with uint8 values [0, 255]
                    or single frame of shape (H, W, 3)

        Returns:
            Preprocessed tensor ready for CLIP
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis]  # Add batch dimension

        processed = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8))
            processed.append(self.preprocess(img))

        return torch.stack(processed)

    def compute_reward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute reward from image embeddings using Equation 2 (basic)
        or Definition 1 (with goal-baseline regularization).

        Args:
            image_embeds: Normalized image embeddings of shape (B, D)

        Returns:
            Rewards of shape (B,)
        """
        g = self.goal_embed  # (1, D)
        s = image_embeds     # (B, D)

        if self.alpha == 0.0 or self.baseline_embed is None:
            # Basic CLIP reward: cosine similarity (Equation 2)
            # Since both are already normalized, dot product = cosine similarity
            rewards = (s * g).sum(dim=-1)
        else:
            # Goal-Baseline Regularization (Definition 1)
            b = self.baseline_embed  # (1, D)

            # Direction from baseline to goal
            direction = g - b  # (1, D)
            direction = F.normalize(direction, dim=-1)

            # Project s onto line L spanned by b and g
            # proj_L(s) = b + <s - b, direction> * direction
            s_proj = b + ((s - b) * direction).sum(dim=-1, keepdim=True) * direction

            # Regularized state embedding
            s_reg = self.alpha * s_proj + (1 - self.alpha) * s

            # Reward = 1 - 0.5 * ||s_reg - g||^2
            rewards = 1.0 - 0.5 * ((s_reg - g) ** 2).sum(dim=-1)

        return rewards

    def reward_from_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        End-to-end: raw frames → rewards.

        Args:
            frames: numpy array of shape (B, H, W, 3) or (H, W, 3)

        Returns:
            rewards as numpy array of shape (B,) or scalar
        """
        single = frames.ndim == 3
        images = self.preprocess_frames(frames)
        embeds = self.encode_images(images)
        rewards = self.compute_reward(embeds)

        result = rewards.cpu().numpy()
        return result.item() if single else result


def compute_epic_distance(
    clip_rewards: np.ndarray,
    human_labels: np.ndarray,
) -> float:
    """
    Compute EPIC distance between CLIP reward model and binary human labels.

    For goal-based tasks, this simplifies to a function of the Pearson
    correlation between the CLIP rewards and the binary labels
    (Lemma 1, Appendix A).

    Args:
        clip_rewards: CLIP reward values for a set of states, shape (N,)
        human_labels: Binary labels (1 = goal state, 0 = not), shape (N,)

    Returns:
        EPIC distance (lower = better; 0 = perfect agreement)
    """
    # Pearson correlation
    rho = np.corrcoef(clip_rewards, human_labels)[0, 1]

    # EPIC distance
    epic = (1 / np.sqrt(2)) * np.sqrt(1 - rho)

    return epic


# ─── Convenience: common CLIP model configurations ────────────────────────────

CLIP_MODELS = {
    "RN50": {"model_name": "RN50", "pretrained": "openai"},
    "ViT-L-14": {"model_name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    "ViT-H-14": {"model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k"},
    "ViT-bigG-14": {"model_name": "ViT-bigG-14", "pretrained": "laion2b_s39b_b160k"},
}


if __name__ == "__main__":
    # Quick test: create a reward model and compute reward for a random image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RN50", choices=CLIP_MODELS.keys(),
                        help="CLIP model to use (RN50 is smallest/fastest for testing)")
    args = parser.parse_args()

    config = CLIP_MODELS[args.model]
    rm = CLIPRewardModel(
        goal_prompt="a humanoid robot kneeling",
        baseline_prompt="a humanoid robot",
        alpha=0.5,
        **config,
    )

    # Test with random noise image
    fake_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    reward = rm.reward_from_frames(fake_frame)
    print(f"Reward for random image: {reward:.4f}")

    # Test batched
    fake_batch = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    rewards = rm.reward_from_frames(fake_batch)
    print(f"Rewards for batch of 4: {rewards}")
