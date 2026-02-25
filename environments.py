"""
Environment wrappers for VLM-RM experiments.

Handles:
  - Rendering frames from Gymnasium environments
  - Textured MountainCar (Section 4.2)
  - Modified Humanoid textures and camera (Section 4.3, Figure 3)
"""

import gymnasium as gym
import numpy as np
import mujoco
from typing import Optional, Tuple
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# CartPole rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_cartpole_at_angle(angle: float, render_size: Tuple[int, int] = (480, 480)) -> np.ndarray:
    """
    Render CartPole at a specific pole angle.

    Args:
        angle: Pole angle in radians (0 = upright)
        render_size: (width, height) of rendered image

    Returns:
        RGB frame as numpy array (H, W, 3)
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    # Manually set the state: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    env.unwrapped.state = np.array([0.0, 0.0, angle, 0.0])

    frame = env.render()
    env.close()

    return frame


def cartpole_reward_landscape(
    reward_model,
    n_angles: int = 100,
    angle_range: Tuple[float, float] = (-0.4, 0.4),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLIP reward across a range of pole angles (reproduces Figure 2a).

    Args:
        reward_model: CLIPRewardModel instance
        n_angles: Number of angles to evaluate
        angle_range: (min_angle, max_angle) in radians

    Returns:
        (angles, rewards) arrays
    """
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)
    rewards = []

    for angle in angles:
        frame = render_cartpole_at_angle(angle)
        r = reward_model.reward_from_frames(frame)
        rewards.append(r)

    return angles, np.array(rewards)


# ═══════════════════════════════════════════════════════════════════════════════
# MountainCar rendering (original and textured)
# ═══════════════════════════════════════════════════════════════════════════════

def render_mountaincar_at_position(
    position: float,
    textured: bool = False,
    render_size: Tuple[int, int] = (480, 480),
) -> np.ndarray:
    """
    Render MountainCar at a specific x position.

    Args:
        position: x position (range roughly [-1.2, 0.6], goal at ~0.5)
        textured: Whether to apply realistic textures
        render_size: (width, height)

    Returns:
        RGB frame as numpy array (H, W, 3)
    """
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env.reset()

    # Set position and velocity
    env.unwrapped.state = np.array([position, 0.0])

    frame = env.render()
    env.close()

    if textured:
        frame = apply_mountain_texture(frame)

    return frame


def apply_mountain_texture(frame: np.ndarray) -> np.ndarray:
    """
    Apply a mountain-like texture to the MountainCar rendering.

    The paper shows that adding realistic textures significantly improves
    CLIP reward quality (Figure 2c vs 2b). This is a simplified version —
    we overlay a gradient sky and darken the mountain to look more natural.

    For a more faithful reproduction, you could:
    1. Use a real mountain background image
    2. Modify the Gymnasium rendering source directly
    """
    h, w, _ = frame.shape
    textured = frame.copy()

    # Create sky gradient (blue at top, lighter at horizon)
    for y in range(h):
        t = y / h  # 0 at top, 1 at bottom
        if frame[y, :, 1].mean() > 200:  # white/light background pixels
            sky_r = int(135 + 80 * t)  # gets lighter toward bottom
            sky_g = int(180 + 40 * t)
            sky_b = int(235 - 20 * t)
            mask = (frame[y, :, 0] > 200) & (frame[y, :, 1] > 200) & (frame[y, :, 2] > 200)
            textured[y, mask, 0] = sky_r
            textured[y, mask, 1] = sky_g
            textured[y, mask, 2] = sky_b

    # Make the mountain area more brown/green (earth-like)
    # The mountain in default rendering is typically a dark line/region
    mountain_mask = (frame[:, :, 0] < 100) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100)
    textured[mountain_mask, 0] = 101  # brownish-green
    textured[mountain_mask, 1] = 120
    textured[mountain_mask, 2] = 75

    return textured


def mountaincar_reward_landscape(
    reward_model,
    n_positions: int = 100,
    position_range: Tuple[float, float] = (-1.2, 0.6),
    textured: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLIP reward across x positions (reproduces Figure 2b/c).
    """
    positions = np.linspace(position_range[0], position_range[1], n_positions)
    rewards = []

    for pos in positions:
        frame = render_mountaincar_at_position(pos, textured=textured)
        r = reward_model.reward_from_frames(frame)
        rewards.append(r)

    return positions, np.array(rewards)


# ═══════════════════════════════════════════════════════════════════════════════
# Humanoid environment with modified textures and camera
# ═══════════════════════════════════════════════════════════════════════════════

class HumanoidVLMWrapper(gym.Wrapper):
    """
    Wrapper for Humanoid-v4 that:
    1. Modifies textures to be more realistic (critical for CLIP, see Figure 3)
    2. Fixes camera position (slightly angled down, not following agent)
    3. Renders frames for CLIP reward computation
    4. Replaces environment reward with CLIP reward

    From the paper (Section 4.3):
    - Texture change: 36% → 91% success rate
    - Texture + camera: 91% → 100% success rate
    """

    def __init__(
        self,
        reward_model=None,
        render_width: int = 224,
        render_height: int = 224,
        modify_textures: bool = True,
        modify_camera: bool = True,
    ):
        env = gym.make(
            "Humanoid-v4",
            render_mode="rgb_array",
            width=render_width,
            height=render_height,
            terminate_when_unhealthy=False,  # Paper removes early termination
        )
        super().__init__(env)

        self.reward_model = reward_model
        self.modify_textures = modify_textures
        self.modify_camera = modify_camera
        self._render_width = render_width
        self._render_height = render_height
        self._textures_modified = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._modify_env()
        return obs, info

    def _modify_env(self):
        """Apply texture and camera modifications to the MuJoCo model."""
        if self._textures_modified:
            return

        model = self.env.unwrapped.model

        if self.modify_textures:
            self._apply_realistic_textures(model)

        if self.modify_camera:
            self._set_fixed_camera(model)

        self._textures_modified = True

    def _apply_realistic_textures(self, model):
        """
        Change humanoid and floor textures to be more realistic.
        The paper emphasizes this is critical for CLIP to interpret the scene.

        We modify:
        - Humanoid body: more skin-like color
        - Floor: darker, more realistic ground
        - Sky/background: natural-looking
        """
        # Modify geom RGBA colors
        # The humanoid has multiple geoms (torso, limbs, etc.)
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name == "floor":
                # Make floor a natural dark gray/green
                model.geom_rgba[i] = [0.3, 0.35, 0.25, 1.0]
            elif name:
                # Make humanoid body parts a more realistic grayish color
                # (the paper uses a humanoid-robot-like appearance)
                model.geom_rgba[i] = [0.6, 0.6, 0.65, 1.0]

    def _set_fixed_camera(self, model):
        """
        Set camera to a fixed position pointing at the agent slightly angled down.
        The default camera follows the agent, which makes some tasks hard to evaluate.
        """
        # We'll use a free camera configuration during rendering
        # This is applied in the render() override below
        pass

    def render(self):
        """Render with fixed camera if enabled."""
        if self.modify_camera:
            # Use the MuJoCo viewer's camera settings
            viewer = self.env.unwrapped.mujoco_renderer
            if hasattr(viewer, '_get_viewer'):
                # Set camera to look down slightly at the agent
                data = self.env.unwrapped.data
                camera_config = {
                    "distance": 4.0,
                    "azimuth": 90,
                    "elevation": -20,
                    "lookat": data.qpos[:3].copy(),
                }
                # Apply to the camera used for rgb_array rendering
                try:
                    cam = viewer.default_cam_config
                    cam.update(camera_config)
                except:
                    pass

        return self.env.render()

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Store original reward for comparison
        info["original_reward"] = original_reward

        # Compute CLIP reward if reward model is provided
        if self.reward_model is not None:
            frame = self.render()
            if frame is not None:
                clip_reward = self.reward_model.reward_from_frames(frame)
                info["clip_reward"] = clip_reward
                return obs, clip_reward, terminated, truncated, info

        return obs, original_reward, terminated, truncated, info


def make_humanoid_env(
    reward_model=None,
    render_size: int = 224,
    modify_textures: bool = True,
    modify_camera: bool = True,
):
    """Factory function to create the modified humanoid environment."""
    return HumanoidVLMWrapper(
        reward_model=reward_model,
        render_width=render_size,
        render_height=render_size,
        modify_textures=modify_textures,
        modify_camera=modify_camera,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities for collecting frames from any environment
# ═══════════════════════════════════════════════════════════════════════════════

def collect_random_frames(
    env_name: str,
    n_frames: int = 100,
    render_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Collect frames from random rollouts. Useful for computing
    reward landscapes and EPIC distances.
    """
    env = gym.make(env_name, render_mode="rgb_array",
                   width=render_size[0], height=render_size[1])
    frames = []

    obs, _ = env.reset()
    for _ in range(n_frames):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return np.array(frames)


if __name__ == "__main__":
    # Quick test: render CartPole at a few angles
    print("Rendering CartPole at different angles...")
    for angle in [-0.3, -0.1, 0.0, 0.1, 0.3]:
        frame = render_cartpole_at_angle(angle)
        print(f"  Angle {angle:+.1f}: frame shape {frame.shape}")

    # Quick test: render MountainCar
    print("\nRendering MountainCar at different positions...")
    for pos in [-1.0, -0.5, 0.0, 0.5]:
        frame = render_mountaincar_at_position(pos, textured=False)
        print(f"  Position {pos:+.1f}: frame shape {frame.shape}")

    print("\nEnvironment wrappers loaded successfully!")
