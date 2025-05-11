import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
    MAIN_ENGINE_POWER,
)
from contextlib import contextmanager

class RocketLander(LunarLander):

    metadata = LunarLander.metadata

    def __init__(
        self,
        enable_wind: bool = False,
        wind_power: float = 0.0,
        turbulence_power: float = 0.0,
        **kwargs,
    ):
        # call the original constructor so we get the whole physics world,
        # reward function, terrain builder, wind generator, etc.
        super().__init__(
            continuous=False,                 
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            **kwargs,
        )

        kwargs.pop("continuous", None)
        self.SCALE = SCALE
        self.VIEWPORT_W = VIEWPORT_W
        self.VIEWPORT_H = VIEWPORT_H
        self.engine_power = MAIN_ENGINE_POWER

        self._ll_action_space = self.action_space          # Discrete(4)

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi/4], dtype=np.float32),
            high=np.array([1.0, np.pi/4], dtype=np.float32),
            dtype=np.float32,
        )

        self._last_thrust = 0.0          # keep for the next render call
        self._last_gimbal = 0.0

        # Nozzle in BODY coordinates (copied from LL source: about 14 px below COM)
        self._nozzle_local = (0.0, -14.0 / self.SCALE)

    def step(self, action):
        if np.isscalar(action):          # called by LL.reset() or legacy code
            thrust = 0.0
            rel_angle = 0.0
        else:
            thrust, rel_angle = np.clip(action, self.action_space.low, self.action_space.high)
            self._last_thrust, self._last_gimbal = float(thrust), float(rel_angle)

        magnitude   = thrust * self.engine_power        # N
        world_angle = self.lander.angle + rel_angle
        fx = -np.sin(world_angle) * magnitude
        fy =  np.cos(world_angle) * magnitude
        self.lander.ApplyForceToCenter((float(fx), float(fy)), True)

        with self._use_ll_action_space():
            obs, reward, terminated, truncated, info = super().step(0)

        return obs, reward, terminated, truncated, info

    
    def reset(self, *, seed: int | None = None, options=None):
        with self._use_ll_action_space():
            obs, info = super().reset(seed=seed, options=options)

        # Move lander horizontally; keep top-of-screen y
        INITIAL_Y = 1.0                        # 1.0 is the max y in LL's coordinates
        rand_x = self.np_random.uniform(-0.5, 0.5)   # tweak range as you like
        self.lander.position = (rand_x, INITIAL_Y)

        # Give the Box2D body a random tilt
        random_angle = self.np_random.uniform(-np.pi/4, np.pi/4)
        self.lander.angle = random_angle
        self.lander.angularVelocity = 0.0

        # Update the returned observation to reflect the new pose
        return obs, info
    
    def render(self, mode=None):
        img = super().render()           # background, terrain, etc.
        
        if not hasattr(self, "viewer") or self.viewer is None:
            return img

        # (1) nozzle world-coords
        cx, cy = self.lander.position
        nx, ny = self._nozzle_local
        sin_theta, cos_theta = np.sin(self.lander.angle), np.cos(self.lander.angle)
        wx = cx + cos_theta * nx - sin_theta * ny
        wy = cy + sin_theta * nx + cos_theta * ny

        # (2) plume tip
        world_angle = self.lander.angle + self._last_gimbal
        length = 30.0 * self._last_thrust / self.SCALE
        ex = wx - np.sin(world_angle) * length
        ey = wy + np.cos(world_angle) * length

        # (3) world → screen
        def w2s(x, y):
            return (
                x * self.SCALE + self.VIEWPORT_W / 2,
                y * self.SCALE + self.VIEWPORT_H / 4,
            )

        sx1, sy1 = w2s(wx, wy)
        sx2, sy2 = w2s(ex, ey)

        # draw / update the polyline
        if not hasattr(self, "_plume"):
            self._plume = self.viewer.draw_polyline(
                [(sx1, sy1), (sx2, sy2)],
                color=(1.0, 0.5, 0.0),
                linewidth=2,
            )
        else:
            self._plume.v = (sx1, sy1, sx2, sy2)   # just overwrite the 4-tuple

        return img

    @contextmanager
    def _use_ll_action_space(self):
        """Temporarily restore the parent Discrete(4) space for its own checks."""
        orig = self.action_space
        self.action_space = self._ll_action_space
        try:
            yield
        finally:
            self.action_space = orig
