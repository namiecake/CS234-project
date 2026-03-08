"""
TEST SCRIPT: Validate humanoid texture/camera setup before full training.

Run from the src/ directory:
    python test_humanoid_setup.py

This script does NOT train anything. It checks:
  1. The textured MuJoCo environment loads without errors
  2. The rendered frame has the right shape and non-trivial pixel values
  3. Visual inspection: saves a rendered frame as PNG so you can confirm
     textures (skybox, tiled floor, metallic robot) and camera angle
  4. CLIP rewards are non-degenerate (varying across a short rollout)
"""

import os
import sys
import numpy as np

PASS = "PASS"
FAIL = "FAIL"


def test_env_creation():
    """Test 1: Can we create the textured environment with camera config?"""
    print("=" * 60)
    print("Test 1: Environment creation")
    print("=" * 60)

    from environments import CLIPRewardedHumanoidEnv, DEFAULT_HUMANOID_CAMERA_CONFIG

    try:
        env = CLIPRewardedHumanoidEnv(
            episode_length=100,
            render_mode="rgb_array",
            camera_config=DEFAULT_HUMANOID_CAMERA_CONFIG,
            textured=True,
            width=480,
            height=480,
        )
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape} (expect (376,))")
        assert obs.shape == (376,), f"Unexpected obs shape: {obs.shape}"

        print(f"  Action space: {env.action_space.shape} (expect (17,))")
        assert env.action_space.shape == (17,), f"Unexpected action shape"

        print(f"  [{PASS}] Environment created successfully.\n")
        return env
    except Exception as e:
        print(f"  [{FAIL}] Environment creation failed: {e}\n")
        raise


def test_rendering(env):
    """Test 2: Does rendering produce a valid frame?"""
    print("=" * 60)
    print("Test 2: Rendering")
    print("=" * 60)

    frame = env.render()
    print(f"  Frame shape: {frame.shape} (expect (480, 480, 3))")
    assert frame.shape == (480, 480, 3), f"Unexpected frame shape: {frame.shape}"
    assert frame.dtype == np.uint8, f"Unexpected dtype: {frame.dtype}"

    mean_val = frame.mean()
    std_val = frame.std()
    print(f"  Pixel mean: {mean_val:.1f}, std: {std_val:.1f}")
    assert std_val > 5.0, "Frame has near-zero variance -- textures may not have loaded"

    print(f"  [{PASS}] Rendering works, frame has non-trivial content.\n")
    return frame


def test_save_frame(frame):
    """Test 3: Save frame for visual inspection."""
    print("=" * 60)
    print("Test 3: Visual inspection (save frame)")
    print("=" * 60)

    from PIL import Image

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_humanoid_textured.png")

    Image.fromarray(frame).save(out_path)
    print(f"  Saved rendered frame to: {out_path}")
    print(f"  >> Open this image and verify:")
    print(f"       - Sky is a photo (NOT a dark gradient)")
    print(f"       - Floor has realistic tiles (NOT a black/white checkerboard)")
    print(f"       - Robot body is shiny/metallic (NOT a flat skin-tone color)")
    print(f"       - Camera views from the front, slightly above")
    print(f"  [{PASS}] Frame saved.\n")


def test_rollout_and_rewards(env):
    """Test 4: Run a few steps and check CLIP rewards vary."""
    print("=" * 60)
    print("Test 4: Short rollout with CLIP reward")
    print("=" * 60)

    from vlm_reward import CLIPRewardModel, CLIP_MODELS

    config = CLIP_MODELS["RN50"]
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    print(f"  Loading RN50 CLIP model on {device} (smallest, for quick test)...")
    rm = CLIPRewardModel(
        goal_prompt="a humanoid robot kneeling",
        baseline_prompt="a humanoid robot",
        alpha=0.0,
        device=device,
        **config,
    )

    rewards = []
    obs, _ = env.reset()
    for step in range(20):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        r = rm.reward_from_frames(frame)
        rewards.append(r)
        if terminated or truncated:
            obs, _ = env.reset()

    rewards = np.array(rewards)
    print(f"  Rewards over 20 random steps:")
    print(f"    mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, max={rewards.max():.4f}")

    assert not np.all(rewards == rewards[0]), \
        "All rewards identical -- reward model may not be receiving distinct frames"
    assert np.all(np.isfinite(rewards)), "Non-finite rewards detected"

    print(f"  [{PASS}] CLIP rewards are finite and varying.\n")


def main():
    print("\n" + "=" * 60)
    print("  HUMANOID TEXTURE/CAMERA SETUP VALIDATION")
    print("  (This is a test script -- it does NOT train anything)")
    print("=" * 60 + "\n")

    env = test_env_creation()
    frame = test_rendering(env)
    test_save_frame(frame)
    test_rollout_and_rewards(env)

    env.close()

    print("=" * 60)
    print(f"  ALL TESTS PASSED")
    print(f"  You are ready to train. Remember to visually inspect")
    print(f"  results/plots/test_humanoid_textured.png before starting.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
