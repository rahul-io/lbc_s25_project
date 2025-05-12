"""rocket_lander_env.py
====================================
Continuous‑action rocket‑landing task using **Gymnasium ≥ 0.29** and its
Pygame renderer (no pyglet viewer required).

Install requirements (Python 3.8–3.12):
```bash
pip install "gymnasium[box2d,classic_control]"
```

Example
-------
```python
import gymnasium as gym
import rocket_lander_env  # this file

env = gym.make("RocketLander-v0", render_mode="human")
obs, _ = env.reset(seed=0)
while True:
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        break
env.close()
```
"""
from __future__ import annotations

import math
from typing import Any, List, Tuple

import numpy as np
from Box2D import b2World  # type: ignore
from Box2D.b2 import (
    distanceJointDef,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

# -----------------------------------------------------------------------------
# Physics / geometry constants
# -----------------------------------------------------------------------------
CONTINUOUS = True
VEL_STATE = True
FPS = 60
SCALE_S = 0.35
INITIAL_RANDOM = 0.4
START_HEIGHT = 800.0
START_SPEED = 80.0

# not worrying about min throttle for now
# MIN_THROTTLE = 0.4
GIMBAL_LIMIT = math.pi/4
MAIN_ENGINE_POWER = 1600 * SCALE_S

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
NOZZLE_LOCAL = (0.0, -ROCKET_HEIGHT / 2)

LEG_LENGTH = ROCKET_WIDTH * 2.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 40

VIEWPORT_H, VIEWPORT_W = 720, 500
H = 1.1 * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return r, g, b


def _w2s(x: float, y: float) -> Tuple[int, int]:
    sx = int(x * (VIEWPORT_W / W))
    sy = int(y * (VIEWPORT_H / H))
    return sx, VIEWPORT_H - sy

# No external collision detection utilities needed - moved to class method

# -----------------------------------------------------------------------------
# Main environment
# -----------------------------------------------------------------------------


class RocketLander(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, *, render_mode: str | None = None):
        EzPickle.__init__(self, render_mode=render_mode)
        self.render_mode = render_mode

        # Box2D world and placeholders
        self.world: b2World = b2World()
        self.water = self.lander = self.ship = None  # type: ignore[assignment]
        self.containers: list[Any] = []
        self.legs: list[Any] = []

        # RNG object assigned in reset
        self.np_random: np.random.Generator

        # Spaces - Updated to remove leg contact variables (2 fewer elements)
        high = np.array([1] * 5 + [np.inf] * 3, dtype=np.float32)  # 5 instead of 7
        low = -high
        if not VEL_STATE:
            high, low = high[:5], low[:5]  # 5 instead of 7
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-GIMBAL_LIMIT, 0.0], dtype=np.float32),
            high=np.array([+GIMBAL_LIMIT, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode vars
        self._prev_shaping: float | None = None
        self._game_over = False
        self._landed_ticks = 0
        self._step_count = 0
        self.throttle = 0.0
        self.gimbal = 0.0
        self.power = 0.0

        # Rendering handles
        self._window = self._surface = None  # created lazily

    # ------------------------------------------------------------------
    # Gymnasium API methods
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

        self._destroy_world()
        self._build_world()

        self._prev_shaping = None
        self._game_over = False
        self._landed_ticks = 0
        self._step_count = 0

        return self._get_state(), {}

    def step(self, action):  # type: ignore[override]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.gimbal   = float(action[0])          # rad (-π/4 … +π/4)
        self.throttle = float(action[1])          # 0 … 1
        self.power = self.throttle

        nozzle_world = self.lander.GetWorldPoint(NOZZLE_LOCAL)
        print("step function")
        print("nozzle_local", NOZZLE_LOCAL)

        # unit vector in engine direction (lander-up rotated by gimbal)
        thrust_dir_world = self.lander.GetWorldVector(
            (-math.sin(self.gimbal), math.cos(self.gimbal))
        )

        print("gimbal (degrees)", math.degrees(self.gimbal))
        
        deg_expected = math.degrees(self.gimbal + self.lander.angle + math.pi/2)
        deg_actual = math.degrees(math.atan2(thrust_dir_world[1], thrust_dir_world[0]))
        print("calculated thrust_dir_world", deg_expected)
        print("thrust_dir_world (degrees)", deg_actual)
        print("diff", (deg_expected - deg_actual + 180) % 360 - 180)
        # print("throttle", self.throttle)

        main_force = (
            thrust_dir_world[0] * MAIN_ENGINE_POWER * self.power,
            thrust_dir_world[1] * MAIN_ENGINE_POWER * self.power,
        )

        # apply at nozzle → produces torque when gimballed
        self.lander.ApplyForce(force=main_force, point=nozzle_world, wake=False)

        # Physics step
        self.world.Step(1.0 / FPS, 60, 60)
        print("nozzle_world", nozzle_world)
        print("lander position", self.lander.position)
        print("lander angle (degrees)", math.degrees(self.lander.angle))
        
        # Check for collision (replaces contact listener)
        if self._check_lander_collision():
            self._game_over = True

        obs = self._get_state()
        reward, terminated = self._compute_reward()
        self._step_count += 1
        return obs, reward, terminated, False, {}
    
    def render(self):  # ← inside class RocketLander
        """
        Pygame renderer supporting `render_mode == "human"` or `"rgb_array"`.
        Creates the window lazily, so importing the env on head-less machines
        will not raise a display error unless you actually call render().
        """
        if self.render_mode is None:          # no-op if rendering disabled
            return None

        import pygame
        
        if not hasattr(self, "_clock"):
            self._clock = pygame.time.Clock()

        # ------------------------------------------------------------------
        # 1.  Lazy-create window / surface
        # ------------------------------------------------------------------
        if self._window is None:
            if self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self._window = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            else:  # "rgb_array"
                self._window = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
            self._surface = self._window
            pygame.display.set_caption("RocketLander-v0")

        # ------------------------------------------------------------------
        # 2.  Clear background
        # ------------------------------------------------------------------
        self._surface.fill(_rgb(126, 150, 233))  # sky-blue

        # Water
        pygame.draw.rect(
            self._surface,
            _rgb(70, 96, 176),
            pygame.Rect(
                0,
                VIEWPORT_H - int(self.terrainheight * VIEWPORT_H / H),
                VIEWPORT_W,
                VIEWPORT_H,
            ),
        )

        # ------------------------------------------------------------------
        # 3.  Ship deck + helipad
        # ------------------------------------------------------------------
        x1, y1 = _w2s(self.helipad_x1, self.terrainheight)
        x2, y2 = _w2s(self.helipad_x2, self.shipheight)
        pygame.draw.rect(
            self._surface,
            _rgb(51, 51, 51),
            pygame.Rect(x1, y2, x2 - x1, y1 - y2),
        )
        # Helipad yellow line
        hx1, hy = _w2s(self.helipad_x1, self.shipheight)
        hx2, _  = _w2s(self.helipad_x2, self.shipheight)
        pygame.draw.line(self._surface, _rgb(206, 206, 2), (hx1, hy), (hx2, hy), 2)

        # ------------------------------------------------------------------
        # 4.  Rocket body (simple rotated rectangle)
        # ------------------------------------------------------------------

        rx, ry = self.lander.position
        rx_s, ry_s = _w2s(rx, ry)

        nx_w, ny_w = self.lander.GetWorldPoint(NOZZLE_LOCAL)
        nx_s,  ny_s    = _w2s(nx_w, ny_w)

        # build upright sprite (centre = COM)
        px_w = int(ROCKET_WIDTH * VIEWPORT_W / W)
        px_h = int(ROCKET_HEIGHT * VIEWPORT_H / H)
        body = pygame.Surface((px_w, px_h), pygame.SRCALPHA)
        body.fill(_rgb(230, 230, 230))

        # rotate it CLOCKWISE by θ → pass -θ to rotozoom
        angle_deg = -math.degrees(self.lander.angle)
        body_rot  = pygame.transform.rotozoom(body, -angle_deg, 1.0)

        # where did the sprite’s nozzle pixel go after rotation?
        import pygame.math as pgm
        offset = pgm.Vector2(0,  px_h / 2).rotate(angle_deg)
        anchor = (nx_s-offset.x, ny_s-offset.y)

        # hang the rotated sprite so its nozzle pixel lands on (nx, ny)
        rect = body_rot.get_rect(center=anchor)
        self._surface.blit(body_rot, rect)

        # ------------------------------------------------------------------
        # 5.  Main-engine flame (line whose length ∝ throttle)
        # ------------------------------------------------------------------
        
        if self.power > 0:
            nozzle_world = self.lander.GetWorldPoint(NOZZLE_LOCAL)
            thrust_dir_world = self.lander.GetWorldVector(
                (-math.sin(self.gimbal), math.cos(self.gimbal))
            )

            # root (screen coords)
            nx, ny = _w2s(*nozzle_world)

            # tip (screen coords) –  scaled version of the same vector
            tip_world = (
                nozzle_world[0] - thrust_dir_world[0] * 1.2 * self.power * ROCKET_HEIGHT,
                nozzle_world[1] - thrust_dir_world[1] * 1.2 * self.power * ROCKET_HEIGHT,
            )
            flame_tip = _w2s(*tip_world)

            # draw flame
            pygame.draw.line(self._surface, _rgb(255, 200, 50), (nx, ny), flame_tip, 4)

        # ------------------------------------------------------------------
        # 6.  Containers (penalty objects at ship edges)
        # ------------------------------------------------------------------
        for container in self.containers:
            # Get the vertices of the container polygon
            vertices = []
            for vertex in container.fixtures[0].shape.vertices:
                world_vertex = container.transform * vertex
                screen_vertex = _w2s(*world_vertex)
                vertices.append(screen_vertex)
            
            # Draw container as a polygon
            if len(vertices) >= 3:  # Need at least 3 points for a polygon
                pygame.draw.polygon(self._surface, _rgb(206, 206, 2), vertices)
        
        # ------------------------------------------------------------------
        # 7.  Legs (visual only)
        # ------------------------------------------------------------------
        for i, leg in enumerate(self.legs):
            # Calculate the leg's attachment point relative to the lander
            side = -1 if i == 0 else 1  # Left leg or right leg
            
            # Calculate leg's position relative to the lander's current position
            # Attach leg to the bottom of the lander
            leg_attach_local = (-side * LEG_AWAY, -ROCKET_HEIGHT/2 + 0.1)
            leg_attach_world = self.lander.GetWorldPoint(leg_attach_local)
            
            # Store the updated world position for the leg
            leg_world_pos = leg_attach_world
            
            # Calculate leg angle (base angle + lander angle)
            leg_angle = side * BASE_ANGLE + self.lander.angle
            
            # Get the vertices of the visual leg
            vertices = []
            for vertex in leg['vertices']:
                # Apply the updated leg position and angle to each vertex
                cos_a = math.cos(leg_angle)
                sin_a = math.sin(leg_angle)
                # Rotate vertex
                rotated_x = vertex[0] * cos_a - vertex[1] * sin_a
                rotated_y = vertex[0] * sin_a + vertex[1] * cos_a
                # Translate to world position
                world_vertex = (
                    leg_world_pos[0] + rotated_x,
                    leg_world_pos[1] + rotated_y,
                )
                screen_vertex = _w2s(*world_vertex)
                vertices.append(screen_vertex)
            
            # Draw leg as a polygon
            if len(vertices) >= 3:  # Need at least 3 points for a polygon
                pygame.draw.polygon(self._surface, leg['color'], vertices)
            
            # Ground contact indicator - show when near the ground
            # Check if any part of the leg is close to the ground
            # Project vertices to world coordinates to check ground contact
            world_vertices = []
            for vertex in leg['vertices']:
                # Apply rotation
                cos_a = math.cos(leg_angle)
                sin_a = math.sin(leg_angle)
                rotated_x = vertex[0] * cos_a - vertex[1] * sin_a
                rotated_y = vertex[0] * sin_a + vertex[1] * cos_a
                # Calculate world position
                world_vertex = (
                    leg_world_pos[0] + rotated_x,
                    leg_world_pos[1] + rotated_y,
                )
                world_vertices.append(world_vertex)
            
            # Find the lowest point of the leg in world coordinates
            leg_bottom_y = min([v[1] for v in world_vertices])
            leg['ground_contact'] = leg_bottom_y <= self.shipheight + 0.1
            
            if leg['ground_contact']:
                # Find the bottom-most point of the leg in screen coordinates
                bottom_point = min(vertices, key=lambda p: p[1])
                pygame.draw.circle(self._surface, _rgb(0, 255, 0), bottom_point, 3)

        # ------------------------------------------------------------------
        # 8.  Present frame / return pixels
        # ------------------------------------------------------------------
        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(30)
            # keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        else:  # "rgb_array"
            return np.transpose(
                np.array(self._surface, copy=False), (1, 0, 2)
            )  # HWC → HWC (already)

    # ------------------------------------------------------------------
    # World-lifecycle helpers
    # ------------------------------------------------------------------
    def _destroy_world(self) -> None:
        """Remove all Box2D bodies from the previous episode (if any)."""
        if self.water is None:                # nothing to clean up
            return

        # No need to destroy legs since they're just visual data now
        for body in [self.water, self.lander, self.ship, *self.containers]:
            self.world.DestroyBody(body)

        # Clear references
        self.water = self.lander = self.ship = None      # type: ignore[assignment]
        self.legs.clear()
        self.containers.clear()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def _build_world(self) -> None:
        """Create water, ship deck, lander, legs and joints for a new episode."""
        # --- Static water rectangle -------------------------------------------------
        self.terrainheight = H / 20                           # flat sea
        water_poly = ((0, 0), (W, 0), (W, self.terrainheight), (0, self.terrainheight))
        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(shape=polygonShape(vertices=water_poly), friction=0.1)
        )
        self.water.color1 = _rgb(70, 96, 176)

        # --- Ship deck --------------------------------------------------------------
        self.shipheight = self.terrainheight + SHIP_HEIGHT
        ship_x = W / 2
        self.helipad_x1 = ship_x - SHIP_WIDTH / 2
        self.helipad_x2 = ship_x + SHIP_WIDTH / 2

        ship_poly = (
            (self.helipad_x1, self.terrainheight),
            (self.helipad_x2, self.terrainheight),
            (self.helipad_x2, self.shipheight),
            (self.helipad_x1, self.shipheight),
        )
        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(shape=polygonShape(vertices=ship_poly), friction=0.5)
        )
        self.ship.color1 = _rgb(51, 51, 51)

        # Containers at the edges (penalty objects)
        self.containers = []
        for side in (-1, 1):
            dx = side * 0.95 * SHIP_WIDTH / 2
            verts = (
                (ship_x + dx, self.shipheight),
                (ship_x + dx, self.shipheight + SHIP_HEIGHT),
                (ship_x + dx - side * SHIP_HEIGHT, self.shipheight + SHIP_HEIGHT),
                (ship_x + dx - side * SHIP_HEIGHT, self.shipheight),
            )
            body = self.world.CreateStaticBody(
                fixtures=fixtureDef(shape=polygonShape(vertices=verts), friction=0.2)
            )
            body.color1 = _rgb(206, 206, 2)
            self.containers.append(body)

        # --- Lander -----------------------------------------------------------------
        spawn_x = W / 2 + W * self.np_random.uniform(-0.3, 0.3)
        spawn_y = H * 0.95
        lander_poly = (
            (-ROCKET_WIDTH / 2, 0),
            (ROCKET_WIDTH / 2, 0),
            (ROCKET_WIDTH / 2, ROCKET_HEIGHT),
            (-ROCKET_WIDTH / 2, ROCKET_HEIGHT),
        )
        self.lander = self.world.CreateDynamicBody(
            position=(spawn_x, spawn_y),
            fixtures=fixtureDef(shape=polygonShape(vertices=lander_poly), density=1.0, friction=0.5),
        )
        self.lander.color1 = _rgb(230, 230, 230)

        # --- Create Visual Legs (no physics) ----------------------------------------
        # Store just enough information to render the legs visually
        self.legs.clear()
        for side in (-1, 1):
            # Create visual leg data without Box2D physics
            visual_leg = {
                'side': side,  # Store which side this leg is on
                'color': _rgb(64, 64, 64),
                'vertices': [
                    (0, 0),
                    (0, LEG_LENGTH / 25),
                    (side * LEG_LENGTH, 0),
                    (side * LEG_LENGTH, -LEG_LENGTH / 20),
                    (side * LEG_LENGTH / 3, -LEG_LENGTH / 7),
                ],
                'ground_contact': False  # Keep this attribute for rendering
            }
            self.legs.append(visual_leg)

        # Initial velocities
        self.lander.linearVelocity = (
            -self.np_random.uniform(0, INITIAL_RANDOM) * START_SPEED * (spawn_x - W / 2) / W,
            -START_SPEED,
        )
        self.lander.angularVelocity = (1 + INITIAL_RANDOM) * self.np_random.uniform(-1, 1)

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        pos = self.lander.position
        vel = np.array(self.lander.linearVelocity) / START_SPEED
        angle = (self.lander.angle / math.pi) % 2
        angle = angle - 2 if angle > 1 else angle

        x_dist = (pos.x - W / 2) / W
        y_dist = (pos.y - self.shipheight) / (H - self.shipheight)

        # State without legs
        state: List[float] = [
            2 * x_dist,
            2 * (y_dist - 0.5),
            angle,
            2 * (self.throttle - 0.5),
            self.gimbal / GIMBAL_LIMIT,
        ]
        if VEL_STATE:
            state.extend([vel[0], vel[1], self.lander.angularVelocity])
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward / termination helper
    # ------------------------------------------------------------------
    def _compute_reward(self) -> Tuple[float, bool]:
        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / START_SPEED
        vel_a = self.lander.angularVelocity

        x_dist = (pos.x - W / 2) / W
        y_dist = (pos.y - self.shipheight) / (H - self.shipheight)
        distance = np.linalg.norm((3 * x_dist, y_dist))           # x weighted more
        speed = np.linalg.norm(vel_l)

        angle = (self.lander.angle / math.pi) % 2
        angle = angle - 2 if angle > 1 else angle

        # Remove leg contact and broken leg checks
        outside = abs(pos.x - W / 2) > W / 2 or pos.y > H
        fuel_cost = 0.1 * (0.5 * self.power) / FPS
        
        # New landing check based on lander position instead of leg contact
        landed = pos.y <= self.shipheight + 0.1 and speed < 0.1 and abs(angle) < 0.1

        reward = -fuel_cost
        terminated = False

        if outside:
            self._game_over = True

        if self._game_over:
            terminated = True
        else:
            shaping = (
                -0.5 * (distance + speed + angle ** 2 + vel_a ** 2)
                # No leg bonus
            )
            if self._prev_shaping is not None:
                reward += shaping - self._prev_shaping
            self._prev_shaping = shaping

            # Landing detection without legs
            if landed:
                self._landed_ticks += 1
            else:
                self._landed_ticks = 0
            if self._landed_ticks >= FPS:
                reward = 1.0
                terminated = True

        if terminated:
            reward += max(-1, 0 - 2 * (speed + distance + abs(angle) + abs(vel_a)))

        return float(np.clip(reward, -1, 1)), terminated

    # ------------------------------------------------------------------
    # Collision detection helper
    # ------------------------------------------------------------------
    def _check_lander_collision(self) -> bool:
        """
        Checks if the lander collides with water or containers.
        Returns True if a collision is detected, False otherwise.
        This replaces the Box2D contact listener for collision detection.
        """
        # Get lander position and dimensions
        pos = self.lander.position
        lander_bottom = pos.y - ROCKET_HEIGHT/2
        lander_left = pos.x - ROCKET_WIDTH/2
        lander_right = pos.x + ROCKET_WIDTH/2

        # Check for water collision - game over if the bottom of the rocket touches water
        if lander_bottom <= self.terrainheight:
            return True
        
        # Check for container collisions using simplified bounding box checks
        for container in self.containers:
            # Get container vertices in world coordinates
            vertices = [container.transform * v for v in container.fixtures[0].shape.vertices]
            
            # Find container bounding box
            container_left = min(v[0] for v in vertices)
            container_right = max(v[0] for v in vertices)
            container_bottom = min(v[1] for v in vertices)
            container_top = max(v[1] for v in vertices)
            
            # Simple rectangular collision check (AABB)
            # This checks if the lander's bounding box overlaps with the container's bounding box
            if (lander_right >= container_left and 
                lander_left <= container_right and
                lander_bottom <= container_top and
                pos.y >= container_bottom):
                return True
        
        # No collision detected
        return False