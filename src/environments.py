"""
Environment wrappers for VLM-RM experiments.

Handles:
  - Rendering frames from Gymnasium environments
  - Textured MountainCar (Section 4.2)
  - Modified Humanoid textures and camera (Section 4.3, Figure 3)
"""

import os
import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
from typing import Any, Dict, Optional, Tuple
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# MountainCar coordinate constants (matching Gymnasium's rendering)
# ═══════════════════════════════════════════════════════════════════════════════

_MC_MIN_X = -1.2
_MC_MAX_X = 0.6
_MC_SCREEN_W = 600
_MC_SCREEN_H = 400
_MC_SCALE = _MC_SCREEN_W / (_MC_MAX_X - _MC_MIN_X)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
_cached_mc_bg: Optional[Image.Image] = None
_cached_mc_car: Optional[Image.Image] = None


def _load_mountain_assets():
    """Load and cache the mountain background and car sprite images."""
    global _cached_mc_bg, _cached_mc_car
    if _cached_mc_bg is None:
        _cached_mc_bg = Image.open(
            os.path.join(_DATA_DIR, 'mountain_car_background.png')
        ).convert('RGBA')
        _cached_mc_car = Image.open(
            os.path.join(_DATA_DIR, 'mountain_car.png')
        ).convert('RGBA')
    return _cached_mc_bg, _cached_mc_car


def _mc_height(x: float) -> float:
    """MountainCar height function (from Gymnasium source)."""
    return np.sin(3 * x) * 0.45 + 0.55


def _mc_slope(x: float) -> float:
    """Derivative of the MountainCar height function."""
    return np.cos(3 * x) * 1.35


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
        textured: Whether to use photorealistic background with car sprite
        render_size: (width, height)

    Returns:
        RGB frame as numpy array (H, W, 3)
    """
    if textured:
        return render_textured_mountaincar(position, render_size=render_size)

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env.reset()
    env.unwrapped.state = np.array([position, 0.0])
    frame = env.render()
    env.close()
    return frame


def render_textured_mountaincar(
    position: float,
    render_size: Tuple[int, int] = (480, 480),
) -> np.ndarray:
    """
    Render MountainCar with a photorealistic mountain background and car sprite.

    Follows the same rendering logic as the original paper's pygame-based
    renderer: the car center is placed at the mathematical surface height
    plus a clearance offset, and rotated to match the slope.
    """
    bg_orig, car_orig = _load_mountain_assets()
    bg = bg_orig.copy()

    clearance = 15
    h = _mc_height(position)
    angle_deg = math.degrees(math.atan(_mc_slope(position)))

    # Car center in final-image pixel coordinates (y=0 at top).
    # Matches the original paper's pygame approach: they draw in a y-up
    # coordinate space at y = clearance + h*scale, then flip the surface.
    cx = (position - _MC_MIN_X) * _MC_SCALE
    cy = _MC_SCREEN_H - (clearance + h * _MC_SCALE)

    car_rotated = car_orig.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

    paste_x = int(cx - car_rotated.width / 2)
    paste_y = int(cy - car_rotated.height / 2)

    bg.paste(car_rotated, (paste_x, paste_y), car_rotated)

    result = bg.convert('RGB')
    if render_size and render_size != (_MC_SCREEN_W, _MC_SCREEN_H):
        result = result.resize(render_size, Image.LANCZOS)
    return np.array(result)


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
# Humanoid environment with realistic textures and camera
# ═══════════════════════════════════════════════════════════════════════════════

# Camera config from the original paper's training config (kube/job.yaml).
# Uses a free camera (camera_id=-1) that tracks the torso body.
DEFAULT_HUMANOID_CAMERA_CONFIG: Dict[str, Any] = {
    "trackbodyid": 1,
    "distance": 3.5,
    "lookat": np.array([0.0, 0.0, 1.0]),
    "elevation": -10.0,
    "azimuth": 180.0,
}


class CLIPRewardedHumanoidEnv(GymHumanoidEnv):
    """
    Humanoid environment matching the original VLM-RM paper implementation.

    Textures are handled via a dedicated MuJoCo XML model (humanoid_textured.xml)
    that references PNG texture files for the skybox, floor, and robot body.
    Camera is configured via MujocoEnv's default_camera_config with a free camera.

    From the paper (Section 4.3):
    - Texture change: 36% -> 91% success rate
    - Texture + camera: 91% -> 100% success rate
    """

    def __init__(
        self,
        episode_length: int = 100,
        render_mode: str = "rgb_array",
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        healthy_z_range: Tuple[float, float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        camera_config: Optional[Dict[str, Any]] = None,
        textured: bool = True,
        **kwargs,
    ):
        terminate_when_unhealthy = False
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            render_mode=render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        if textured:
            model_path = os.path.join(_DATA_DIR, "humanoid_textured.xml")
        else:
            model_path = "humanoid.xml"

        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        self.episode_length = episode_length
        self.num_steps = 0
        if camera_config:
            self.camera_id = -1

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)


class HumanoidVLMWrapper(gym.Wrapper):
    """
    Wrapper around CLIPRewardedHumanoidEnv that replaces the environment
    reward with CLIP reward when a reward_model is provided.
    """

    def __init__(
        self,
        reward_model=None,
        render_width: int = 224,
        render_height: int = 224,
        textured: bool = True,
        camera_config: Optional[Dict[str, Any]] = None,
        episode_length: int = 100,
    ):
        if camera_config is None:
            camera_config = DEFAULT_HUMANOID_CAMERA_CONFIG

        env = CLIPRewardedHumanoidEnv(
            episode_length=episode_length,
            render_mode="rgb_array",
            camera_config=camera_config,
            textured=textured,
            width=render_width,
            height=render_height,
        )
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = original_reward

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
    textured: bool = True,
    camera_config: Optional[Dict[str, Any]] = None,
    episode_length: int = 100,
):
    """Factory function to create the modified humanoid environment."""
    return HumanoidVLMWrapper(
        reward_model=reward_model,
        render_width=render_size,
        render_height=render_size,
        textured=textured,
        camera_config=camera_config,
        episode_length=episode_length,
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
