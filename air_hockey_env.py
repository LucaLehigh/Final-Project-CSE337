import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AirHockeyEnv(gym.Env):
    """
    Simplified 2D air-hockey environment.

    State (8D):
        [x_puck, y_puck, vx_puck, vy_puck,
         x_paddle, y_paddle, vx_paddle, vy_paddle]

    Action (2D continuous):
        [ax_paddle, ay_paddle]  (desired paddle acceleration, scaled)

    Table coordinates:
        x in [-TABLE_W/2, TABLE_W/2]
        y in [-TABLE_H/2, TABLE_H/2]

    Episode ends when:
        - max_steps reached
        - puck crosses a "goal line" (optional for now)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        dt: float = 0.02,
        table_width: float = 2.0,
        table_height: float = 1.0,
        paddle_radius: float = 0.07,
        puck_radius: float = 0.05,
        max_paddle_speed: float = 3.0,
        puck_friction: float = 0.99,
        max_steps: int = 500,
        # ||WIP|| dynamics randomization not fully implemented
        randomize_dynamics: bool = False,
    ):
        super().__init__()

        # Simulation params
        self.dt = dt
        self.TABLE_W = table_width
        self.TABLE_H = table_height
        self.PADDLE_R = paddle_radius
        self.PUCK_R = puck_radius
        self.MAX_PADDLE_SPEED = max_paddle_speed
        self.PUCK_FRICTION = puck_friction
        self.max_steps = max_steps

        # ||WIP|| flags and ranges
        self.randomize_dynamics = randomize_dynamics
        # TODO: tune these ranges, currently just placeholders
        self._friction_range = (0.95, 0.999)
        self._restitution_range = (0.5, 1.1)  # might explode if > 1
        self._tilt_range = (-0.1, 0.1)  # "gravity" in x/y but not used yet

        # This restitution is never respected in step() yet (still hard-coded there)
        self.restitution = 0.9

        # Spaces
        # Action: paddle acceleration in x,y ∈ [-1,1], scaled inside step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation bounds (loose but finite to satisfy Gym)
        high_pos = np.array(
            [
                self.TABLE_W / 2,
                self.TABLE_H / 2,
                np.inf,
                np.inf,
                self.TABLE_W / 2,
                self.TABLE_H / 2,
                self.MAX_PADDLE_SPEED,
                self.MAX_PADDLE_SPEED,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high_pos, high=high_pos, dtype=np.float32
        )

        self.render_mode = render_mode
        self._state = None
        self._steps = 0
        self._rng = np.random.default_rng()

    # Gym API: reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0

        # ||WIP|| this tries to randomize physics but is not wired correctly
        self.randomize_physics()

        # Initialize puck near center with small random velocity
        x_puck = self._rng.uniform(-0.1, 0.1)
        y_puck = self._rng.uniform(-0.1, 0.1)
        vx_puck = self._rng.uniform(-0.5, 0.5)
        vy_puck = self._rng.uniform(-0.5, 0.5)

        # Initialize paddle near "bottom" (e.g., negative y side)
        x_paddle = 0.0
        y_paddle = -self.TABLE_H / 4
        vx_paddle = 0.0
        vy_paddle = 0.0

        self._state = np.array(
            [
                x_puck,
                y_puck,
                vx_puck,
                vy_puck,
                x_paddle,
                y_paddle,
                vx_paddle,
                vy_paddle,
            ],
            dtype=np.float32,
        )

        obs = self._get_obs()
        info = {}
        return obs, info

    def randomize_physics(self):
        """
        WIP:
        Randomize friction, restitution, and a fake table tilt.
        Not fully integrated with the rest of the physics yet.
        """
        if not getattr(self, "randomize_dynamics", False):
            return

        # Randomize friction (this one *does* get used)
        low, high = self._friction_range
        self.PUCK_FRICTION = float(self._rng.uniform(low, high))

        # Randomize restitution, but step() still uses a hard-coded value
        r_low, r_high = self._restitution_range
        self.restitution = float(self._rng.uniform(r_low, r_high))

        # TODO: actually apply tilt to puck motion in step()
        tx = self._rng.uniform(*self._tilt_range)
        ty = self._rng.uniform(*self._tilt_range)
        # Currently unused, just stored
        self.table_tilt = np.array([tx, ty], dtype=np.float32)
        # NOTE: table_tilt is never applied in the dynamics, so this is incomplete.

    # Gym API: step
    def step(self, action):
        """
        One simulation step with simple kinematics and collisions.
        """
        self._steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        ax, ay = action.astype(np.float32)

        # Unpack state
        x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad = self._state

        # Update paddle velocity + position
        # Scale acceleration so [-1,1] is reasonable
        PAD_ACC_SCALE = 10.0
        vx_pad += PAD_ACC_SCALE * ax * self.dt
        vy_pad += PAD_ACC_SCALE * ay * self.dt

        # Clamp paddle speed
        speed = np.sqrt(vx_pad**2 + vy_pad**2)
        if speed > self.MAX_PADDLE_SPEED:
            vx_pad *= self.MAX_PADDLE_SPEED / (speed + 1e-8)
            vy_pad *= self.MAX_PADDLE_SPEED / (speed + 1e-8)

        # Integrate paddle position
        x_pad += vx_pad * self.dt
        y_pad += vy_pad * self.dt

        # Constrain paddle within table (simple clipping)
        x_pad = np.clip(
            x_pad, -self.TABLE_W / 2 + self.PADDLE_R, self.TABLE_W / 2 - self.PADDLE_R
        )
        y_pad = np.clip(
            y_pad, -self.TABLE_H / 2 + self.PADDLE_R, self.TABLE_H / 2 - self.PADDLE_R
        )

        # --- Update puck position with friction ---
        x_p += vx_p * self.dt
        y_p += vy_p * self.dt

        vx_p *= self.PUCK_FRICTION
        vy_p *= self.PUCK_FRICTION

        # WIP: try to apply table tilt as a fake gravity term.
        # if hasattr(self, "table_tilt"):
        #     vx_p += self.table_tilt[0] * self.dt  # TODO: check sign & scaling
        #     vy_p += self.table_tilt[1] * self.dt  # TODO: clamp velocities?

        # Collisions: puck with table walls
        # Left/right walls
        if x_p - self.PUCK_R < -self.TABLE_W / 2:
            x_p = -self.TABLE_W / 2 + self.PUCK_R
            vx_p = -vx_p
        elif x_p + self.PUCK_R > self.TABLE_W / 2:
            x_p = self.TABLE_W / 2 - self.PUCK_R
            vx_p = -vx_p

        # Top/bottom walls
        if y_p - self.PUCK_R < -self.TABLE_H / 2:
            y_p = -self.TABLE_H / 2 + self.PUCK_R
            vy_p = -vy_p
        elif y_p + self.PUCK_R > self.TABLE_H / 2:
            y_p = self.TABLE_H / 2 - self.PUCK_R
            vy_p = -vy_p

        # Collision: puck with paddle
        dx = x_p - x_pad
        dy = y_p - y_pad
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = self.PUCK_R + self.PADDLE_R

        if dist < min_dist:
            # Simple elastic-ish collision: reflect puck velocity along normal
            # Normal from paddle -> puck
            nx = dx / (dist + 1e-8)
            ny = dy / (dist + 1e-8)

            # Relative velocity puck - paddle
            rvx = vx_p - vx_pad
            rvy = vy_p - vy_pad

            # Project relative velocity onto normal
            rel_normal_speed = rvx * nx + rvy * ny

            # Only reflect if moving toward the paddle
            if rel_normal_speed < 0.0:
                restitution = self.restitution  # or 0.9 if you want it hard-coded
                # Impulse magnitude along normal (simplified, unit masses)
                j = -(1.0 + restitution) * rel_normal_speed

                vx_p += j * nx
                vy_p += j * ny

                # Separate overlapping bodies
                overlap = min_dist - dist
                x_p += nx * overlap
                y_p += ny * overlap

        # Pack new state
        self._state = np.array(
            [x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad],
            dtype=np.float32,
        )

        obs = self._get_obs()

        # For Week 1, keep reward minimal; we’ll refine in Week 3.
        # For now: small step penalty to encourage shorter episodes.
        reward = -0.001

        # Episode termination: for now, just max_steps
        terminated = False
        truncated = self._steps >= self.max_steps
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # Helper Functions
    def _get_obs(self):
        return self._state.copy()

    def render(self):
        """
        Minimal placeholder. In Week 1 you can leave this as a stub,
        or later add a matplotlib visualization.
        """
        # You can implement a plot or print occasionally if you want.
        pass

    def close(self):
        pass
