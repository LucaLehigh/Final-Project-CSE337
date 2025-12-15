import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class AirHockeyMujocoEnv(gym.Env):
    """
    Air hockey in MuJoCo.

    State (8D):
        [x_puck, y_puck, vx_puck, vy_puck,
         x_paddle, y_paddle, vx_paddle, vy_paddle]

    Action (2D):
        [ax, ay] in [-1, 1] mapped to paddle motors.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        xml_path="air_hockey.xml",
        max_steps: int = 500,
        # NEW: physics randomization flags
        randomize_gravity: bool = False,
        randomize_friction: bool = False,
        randomize_restitution: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._steps = 0

        self.randomize_gravity = randomize_gravity
        self.randomize_friction = randomize_friction
        self.randomize_restitution = randomize_restitution

        # Load MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # For video / visualization
        self.renderer = None
        if render_mode in ("rgb_array", "human"):
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)

        # Indices for joints
        jnt = self.model.joint
        self.puck_free_id = jnt("puck_free").id
        self.paddle_x_id = jnt("paddle_slide_x").id
        self.paddle_y_id = jnt("paddle_slide_y").id

        # Geoms
        geom = self.model.geom
        self.puck_geom_id = geom("puck_geom").id
        self.paddle_geom_id = geom("paddle_geom").id
        self.table_geom_id = geom("table").id
        self.wall_geom_ids = [
            geom("wall_left").id,
            geom("wall_right").id,
            geom("wall_top").id,
            geom("wall_bottom").id,
        ]

        # Save baseline physics to randomize around
        self.base_gravity = self.model.opt.gravity.copy()
        self.base_table_friction = self.model.geom_friction[self.table_geom_id].copy()
        self.base_wall_friction = {
            gid: self.model.geom_friction[gid].copy() for gid in self.wall_geom_ids
        }
        self.base_wall_solref = {
            gid: self.model.geom_solref[gid].copy() for gid in self.wall_geom_ids
        }

        # previous puck vy for delta_vy reward
        self.prev_vy_p = 0.0

        # qpos & qvel addresses
        self.puck_qpos_adr = self.model.jnt_qposadr[self.puck_free_id]  # 7 values
        self.puck_qvel_adr = self.model.jnt_dofadr[self.puck_free_id]  # 6 values

        self.paddle_x_qpos_adr = self.model.jnt_qposadr[self.paddle_x_id]
        self.paddle_x_qvel_adr = self.model.jnt_dofadr[self.paddle_x_id]
        self.paddle_y_qpos_adr = self.model.jnt_qposadr[self.paddle_y_id]
        self.paddle_y_qvel_adr = self.model.jnt_dofadr[self.paddle_y_id]

        # Action space: ax, ay in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Paddle workspace limits
        self.paddle_x_min = -0.72
        self.paddle_x_max = 0.72
        self.paddle_y_min = -0.2  # close to our goal
        self.paddle_y_max = 0.7  # just before midline

        # Scale used to normalize obs
        self.obs_scale = np.array(
            [0.75, 1.0, 10.0, 10.0, 0.75, 1.0, 10.0, 10.0],
            dtype=np.float32,
        )

        # Observation space: normalized to roughly [-1,1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

    # Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)

        # ===== Physics randomization (per-episode) =====
        # Gravity: random in x,y, mostly downward in z
        if self.randomize_gravity:
            # random small sideways components
            gx = rng.uniform(-2.0, 2.0)
            gy = rng.uniform(-2.0, 2.0)

            # draw a downward magnitude
            gz_mag = rng.uniform(5.0, 12.0)  # how strong gravity is

            # ensure downward component dominates sideways ones
            max_xy = max(abs(gx), abs(gy))
            if gz_mag < max_xy + 1.0:
                gz_mag = max_xy + 1.0

            gz = -gz_mag  # downward

            self.model.opt.gravity[:] = np.array([gx, gy, gz], dtype=np.float64)
        else:
            # reset to default gravity
            self.model.opt.gravity[:] = self.base_gravity

        # Friction on table + walls
        if self.randomize_friction:
            # table
            tf = self.model.geom_friction[self.table_geom_id]
            base_tf = self.base_table_friction
            tf[0] = base_tf[0] * rng.uniform(0.5, 2.0)  # sliding friction
            # walls
            for gid in self.wall_geom_ids:
                wf = self.model.geom_friction[gid]
                base_wf = self.base_wall_friction[gid]
                wf[0] = base_wf[0] * rng.uniform(0.5, 2.0)
        else:
            self.model.geom_friction[self.table_geom_id] = self.base_table_friction
            for gid in self.wall_geom_ids:
                self.model.geom_friction[gid] = self.base_wall_friction[gid]

        # "Bounciness" via solref on walls
        if self.randomize_restitution:
            for gid in self.wall_geom_ids:
                sr = self.model.geom_solref[gid]
                base_sr = self.base_wall_solref[gid]
                sr[0] = base_sr[0] * rng.uniform(0.5, 1.5)  # time constant
                sr[1] = base_sr[1] * rng.uniform(0.8, 1.2)  # damping
        else:
            for gid in self.wall_geom_ids:
                self.model.geom_solref[gid] = self.base_wall_solref[gid]
        # ===============================================

        self.prev_dist = None

        x_p = rng.uniform(-0.7, 0.7)
        y_p = rng.uniform(0.4, 0.8)

        vx0 = rng.uniform(-1.0, 1.0)
        vy0 = -rng.uniform(0.5, 1.5)

        # Apply to MuJoCo (puck)
        self.data.qpos[self.puck_qpos_adr + 0] = x_p
        self.data.qpos[self.puck_qpos_adr + 1] = y_p
        self.data.qpos[self.puck_qpos_adr + 2] = 0.02
        self.data.qpos[self.puck_qpos_adr + 3] = 1.0
        self.data.qpos[self.puck_qpos_adr + 4 : self.puck_qpos_adr + 7] = 0.0

        self.data.qvel[self.puck_qvel_adr + 0] = vx0
        self.data.qvel[self.puck_qvel_adr + 1] = vy0
        self.data.qvel[self.puck_qvel_adr + 2] = 0.0
        self.data.qvel[self.puck_qvel_adr + 3 : self.puck_qvel_adr + 6] = 0.0

        # Paddle starting pose
        self.data.qpos[self.paddle_x_qpos_adr] = 0.0
        self.data.qpos[self.paddle_y_qpos_adr] = 0.0
        self.data.qvel[self.paddle_x_qvel_adr] = 0.0
        self.data.qvel[self.paddle_y_qvel_adr] = 0.0

        # Clear controls
        self.data.ctrl[:] = 0.0

        obs = self._get_obs()
        self.prev_vy_p = obs[3] * self.obs_scale[3]  # restore actual vy for delta_vy
        info = {}
        return obs, info

    def step(self, action):
        self._steps += 1

        # action & controls
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ax, ay = action.astype(np.float32)

        MAX_FORCE = 1.5
        self.data.ctrl[0] = ax * MAX_FORCE
        self.data.ctrl[1] = ay * MAX_FORCE

        # physics step
        frame_skip = 5
        for _ in range(frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.data.qpos[self.puck_qpos_adr + 2] = 0.02  # fixed height above table
        self.data.qvel[self.puck_qvel_adr + 2] = 0.0  # no vertical velocity
        # kill angular velocity as well (we don't use rotation in obs)
        self.data.qvel[self.puck_qvel_adr + 3 : self.puck_qvel_adr + 6] = 0.0

        vx_pad = float(self.data.qvel[self.paddle_x_qvel_adr])
        vy_pad = float(self.data.qvel[self.paddle_y_qvel_adr])
        speed = np.sqrt(vx_pad**2 + vy_pad**2)
        V_MAX = 1.0  # tune: lower = slower, safer contacts

        if speed > V_MAX:
            scale = V_MAX / speed
            self.data.qvel[self.paddle_x_qvel_adr] *= scale
            self.data.qvel[self.paddle_y_qvel_adr] *= scale

        # clamp paddle to its box on our half
        px = float(self.data.qpos[self.paddle_x_qpos_adr])
        py = float(self.data.qpos[self.paddle_y_qpos_adr])

        px_clamped = np.clip(px, self.paddle_x_min, self.paddle_x_max)
        py_clamped = np.clip(py, self.paddle_y_min, self.paddle_y_max)

        if px_clamped != px:
            self.data.qpos[self.paddle_x_qpos_adr] = px_clamped
            self.data.qvel[self.paddle_x_qvel_adr] = 0.0
        if py_clamped != py:
            self.data.qpos[self.paddle_y_qpos_adr] = py_clamped
            self.data.qvel[self.paddle_y_qvel_adr] = 0.0

        # Build observation from the state (normalized)
        obs = self._get_obs()
        x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad = obs * self.obs_scale

        # Dynamic puck color: yellow in our half (y<0), red in opponent half (y>=0)
        if y_p < 0.0:
            self.model.geom_rgba[self.puck_geom_id] = np.array([1.0, 1.0, 0.0, 1.0])
        else:
            self.model.geom_rgba[self.puck_geom_id] = np.array([1.0, 0.1, 0.0, 1.0])

        # Reward shaping
        dx = x_p - x_pad
        dy = y_p - y_pad
        dist = np.sqrt(dx**2 + dy**2)

        reward = 0.0
        scored = False
        conceded = False

        # Goal events
        # goal_half_width = 0.2  # half of 0.4 total
        # in_goal_x = abs(x_p) < goal_half_width

        # if in_goal_x and (y_p > 0.9) and (vy_p > 0):
        #     scored = True
        # if in_goal_x and (y_p < -0.9) and (vy_p < 0):
        #     conceded = True

        # full-width goals along the wall
        if (y_p > 0.9) and (vy_p > 0):
            scored = True
        if (y_p < -0.9) and (vy_p < 0):
            conceded = True

        # ---------- Simple "just hit the puck upward" reward ----------

        if scored:
            reward += 10.0  # big positive
        if conceded:
            reward -= 8.0  # big negative

        # --- TWO-MODE SHAPING: based on puck half ---

        paddle_base_y = 0.0

        if y_p < 0.0:
            # OUR HALF & puck coming toward us
            if vy_p < 0.0:
                max_dist = np.sqrt(0.75**2 + 1.0**2)
                dist_n = np.clip(dist / max_dist, 0.0, 1.0)
                proximity = 1.0 - dist_n
                reward += 0.1 * proximity

                if y_pad < y_p:
                    reward += 0.005 * (y_p - y_pad)
        else:
            # OPPONENT HALF: drift back to base
            base_err = y_pad - paddle_base_y
            reward -= 0.03 * (base_err**2)

            if y_pad > 0.0:
                reward -= 0.05

        # --- Contact-based shaping: send puck upward when hitting it ---
        contact_radius = 0.15
        if dist < contact_radius:
            delta_vy = vy_p - getattr(self, "prev_vy_p", 0.0)

            if delta_vy > 0.0:
                reward += 0.3 * np.clip(delta_vy / 5.0, 0.0, 1.0)
            else:
                reward += 0.03

        # Time penalty
        reward -= 0.001

        # Remember current vy (un-normalized)
        self.prev_vy_p = vy_p

        # Clip reward
        reward = np.clip(reward, -10.0, 10.0)

        terminated = scored or conceded
        truncated = self._steps >= self.max_steps
        info = {"scored": scored, "conceded": conceded}

        if self.render_mode == "human" and self.renderer is not None:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Return normalized 8D observation."""
        # Puck
        x_p = self.data.qpos[self.puck_qpos_adr + 0]
        y_p = self.data.qpos[self.puck_qpos_adr + 1]
        vx_p = self.data.qvel[self.puck_qvel_adr + 0]
        vy_p = self.data.qvel[self.puck_qvel_adr + 1]

        # Paddle
        x_pad = self.data.qpos[self.paddle_x_qpos_adr]
        y_pad = self.data.qpos[self.paddle_y_qpos_adr]
        vx_pad = self.data.qvel[self.paddle_x_qvel_adr]
        vy_pad = self.data.qvel[self.paddle_y_qvel_adr]

        obs = np.array(
            [x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad],
            dtype=np.float32,
        )

        return obs / self.obs_scale

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.001)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
