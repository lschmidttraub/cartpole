import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces


class CartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load MuJoCo model
        with open("envs/cartpole.xml", "r") as f:
            xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        obs_high = np.array([1.0, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: continuous force on cart [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Add small random perturbation
        if seed is not None:
            self.data.qpos += self.np_random.uniform(-0.01, 0.01, size=self.data.qpos.shape)
            self.data.qvel += self.np_random.uniform(-0.01, 0.01, size=self.data.qvel.shape)

        return self._get_obs(), {}

    def step(self, action):
        # Apply force to cart
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        cart_pos, _, pole_angle, _ = obs

        # Termination conditions
        terminated = bool(
            abs(cart_pos) > 1.0  # Cart out of bounds
            or abs(pole_angle) > 0.5  # Pole fell (~28 degrees)
        )

        # Reward: +1 for staying alive
        reward = 1.0 if not terminated else 0.0

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],  # cart position
            self.data.qvel[0],  # cart velocity
            self.data.qpos[1],  # pole angle
            self.data.qvel[1],  # pole angular velocity
        ], dtype=np.float32)

    def render(self):
        if self.render_mode == "human" and self.renderer:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        if self.renderer:
            self.renderer.close()

