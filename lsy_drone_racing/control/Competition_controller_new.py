"""Spline-following race controller.

Attempts to fly through four gates in sequence using cubic spline interpolation
for smooth trajectory generation.  PID tracking converts position / velocity
errors into attitude + thrust commands.  Obstacle proximity triggers recursive
waypoint insertion until a safe path is found.

Tuned for Level 2 (randomised inertia + gate / obstacle positions).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicSpline as Spline3
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class CompetitionController(Controller):
    """Cubic-spline race controller with per-leg PID and obstacle dodge."""

    __slots__ = (
        "_hz", "_mass", "_gravity",
        "_gate_yaw", "_gate_pos", "_obs_pos",
        "_pid", "_int_clamp", "_int_err",
        "_leg", "_leg_start", "_prev_leg",
        "_traj", "_horizon", "_step", "_finished",
        "_need_plan", "_last_cmd", "_anchor", "_dodge_iter",
    )

    # time budget per leg (seconds)
    LEG_DURATIONS = np.array([2.5, 2.5, 2.5, 2.2])
    DODGE_LIMIT = 5
    SAFE_RADIUS = 0.15
    PROBE_DENSITY = 35

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._hz = config.env.freq

        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = params["mass"]
        self._gravity = 9.81

        # initial track layout (overwritten from sensor data each step)
        self._gate_yaw = np.array([-0.78, 2.35, math.pi, 0.0])
        self._gate_pos = np.array([
            [0.5, 0.25, 0.7], [1.05, 0.75, 1.2],
            [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2],
        ])
        self._obs_pos = np.array([
            [0.0, 0.75, 1.55], [1.0, 0.25, 1.55],
            [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55],
        ])

        # PID tensor: axis 0 = P/I/D, axis 1 = leg, axis 2 = xyz
        self._pid = np.array([
            [[0.6, 0.6, 1.55], [0.45, 0.45, 1.55],
             [0.6, 0.6, 1.55], [0.55, 0.55, 1.55]],
            [[0.05, 0.05, 0.05], [0.045, 0.045, 0.05],
             [0.045, 0.045, 0.05], [0.05, 0.05, 0.05]],
            [[0.35, 0.35, 0.5], [0.35, 0.35, 0.5],
             [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        ])
        self._int_clamp = np.array([2.0, 2.0, 0.4])
        self._int_err = np.zeros(3)

        self._leg = -1
        self._leg_start = np.zeros(4)
        self._prev_leg = -1
        self._traj = None
        self._horizon = 25.0
        self._step = 0
        self._finished = False
        self._need_plan = True
        self._last_cmd = np.zeros(4, dtype=np.float32)
        self._anchor = None
        self._dodge_iter = 0

    # ---- main entry point ------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None,
    ) -> NDArray[np.floating]:
        t = min(self._step / self._hz, self._horizon)
        if t >= self._horizon:
            self._finished = True

        leg = obs["target_gate"]
        gp = obs["gates_pos"]
        op = obs["obstacles_pos"]
        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"]

        # refresh gate headings from quaternion data
        self._gate_yaw = Rot.from_quat(obs["gates_quat"]).as_euler("xyz")[:, 2]

        if self._leg != leg:
            self._leg = leg
            self._int_err[:] = 0.0

        if leg == -1:
            self._finished = True
            return self._last_cmd

        # decide whether trajectory needs a rebuild
        gate_shifted = not np.allclose(self._gate_pos[leg], gp[leg], atol=0.01)
        obs_changed = not np.array_equal(self._obs_pos, op)
        leg_changed = self._prev_leg != leg

        if self._need_plan or gate_shifted or leg_changed or obs_changed:
            self._obs_pos = op.copy()
            self._gate_pos = gp.copy()
            self._traj = self._build_trajectory(t, leg, gp, pos, vel)
            self._need_plan = False

        self._last_cmd = self._track(t, pos, vel, quat)
        return self._last_cmd

    # ---- PID position tracker --------------------------------------------

    def _track(self, t, pos, vel, quat):
        ref_pos = self._traj(t)
        ref_vel = self._traj.derivative()(t)

        e_pos = ref_pos - pos
        e_vel = ref_vel - vel

        self._int_err = np.clip(
            self._int_err + e_pos / self._hz,
            -self._int_clamp, self._int_clamp,
        )

        k = self._leg
        force = (
            self._pid[0, k] * e_pos
            + self._pid[1, k] * self._int_err
            + self._pid[2, k] * e_vel
        )
        force[2] += self._mass * self._gravity

        return self._force_to_cmd(force, quat)

    @staticmethod
    def _force_to_cmd(force, quat):
        """Convert desired force vector into [roll, pitch, yaw, thrust]."""
        body_z = Rot.from_quat(quat).as_matrix()[:, 2]
        thrust = float(force @ body_z)

        z_hat = force / np.linalg.norm(force)
        y_hat = np.cross(z_hat, [1.0, 0.0, 0.0])
        y_hat /= np.linalg.norm(y_hat)
        x_hat = np.cross(y_hat, z_hat)

        rpy = Rot.from_matrix(np.column_stack([x_hat, y_hat, z_hat])).as_euler("xyz")
        return np.array([*rpy, thrust], dtype=np.float32)

    # ---- trajectory construction -----------------------------------------

    def _build_trajectory(self, t, leg, gates, pos, vel, extra_wp=None):
        """Fit a spline through leg waypoints, then recursively dodge obstacles."""
        pts = self._leg_waypoints(leg, gates, extra_wp)
        t0, t1 = self._leg_interval(leg, t)
        traj = Spline3(np.linspace(t0, t1, len(pts)), pts)

        collision_wp = self._probe_obstacles(traj, t0, t1, leg)
        if collision_wp is not None and self._dodge_iter < self.DODGE_LIMIT:
            self._dodge_iter += 1
            traj = self._build_trajectory(t, leg, gates, pos, vel, collision_wp)
            self._dodge_iter = 0

        return traj

    def _leg_interval(self, leg, t):
        if self._prev_leg != leg:
            self._leg_start[leg] = t
            self._prev_leg = leg
        start = self._leg_start[leg]
        return start, start + self.LEG_DURATIONS[leg]

    def _leg_waypoints(self, leg, g, extra):
        """Dispatch to per-leg waypoint builder."""
        match leg:
            case 0: return self._pts_launch(g, extra)
            case 1: return self._pts_cross(g, extra)
            case 2: return self._pts_return(g, extra)
            case _: return self._pts_finish(g, extra)

    # ---- per-leg waypoint sets -------------------------------------------

    def _pts_launch(self, g, extra):
        _, after = self._through_gate(g[0], self._gate_yaw[0], 0.4, 0.4)
        after[2] = g[0][2] + 0.2
        self._anchor = after
        pts = [np.array([-1.5, 0.75, 0.05])]
        if extra is not None:
            pts.append(extra)
        pts += [g[0], after]
        return np.array(pts)

    def _pts_cross(self, g, extra):
        pts = [g[0], np.array([1.2, 0.0, 1.1])]
        if extra is not None:
            pts.append(extra)
        pts += [g[1], np.array([-0.5, -0.05, 0.8])]
        return np.array(pts)

    def _pts_return(self, g, extra):
        _, g1_out = self._through_gate(g[1], self._gate_yaw[1], 0.1, 0.2)
        _, g2_out = self._through_gate(g[2], self._gate_yaw[2], 0.4, 0.3)
        pts = [g1_out]
        if extra is not None:
            pts.append(extra)
        pts += [np.array([-0.5, -0.05, 0.8]), g[2], g2_out]
        return np.array(pts)

    def _pts_finish(self, g, extra):
        _, g3_out = self._through_gate(g[3], self._gate_yaw[3], 0.2, 0.4)
        pts = [g[2], np.array([-0.5, -0.4, 0.9])]
        if extra is not None:
            pts.append(extra)
        pts += [g[3], g3_out]
        return np.array(pts)

    # ---- obstacle proximity check (vectorised) ---------------------------

    def _probe_obstacles(self, traj, t0, t1, leg):
        """Check trajectory for obstacle proximity; return dodge waypoint or None."""
        oi = min(leg, len(self._obs_pos) - 1)
        pole_xy = self._obs_pos[oi, :2]

        samples = np.linspace(t0, t1, self.PROBE_DENSITY)
        pts = traj(samples)                       # (N, 3)
        deltas = pole_xy - pts[:, :2]             # (N, 2)
        dists = np.linalg.norm(deltas, axis=1)    # (N,)

        closest = int(np.argmin(dists))
        if dists[closest] >= self.SAFE_RADIUS:
            return None

        d_vec = deltas[closest]
        push = d_vec / (dists[closest] + 1e-6) * self.SAFE_RADIUS
        return np.array([
            self._obs_pos[oi, 0] - push[0],
            self._obs_pos[oi, 1] - push[1],
            pts[closest, 2],
        ])

    # ---- geometry utilities ----------------------------------------------

    @staticmethod
    def _through_gate(center, yaw, d_before=0.2, d_after=0.4):
        """Points just before and after a gate along its heading."""
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        return center - d_before * heading, center + d_after * heading

    # ---- framework hooks -------------------------------------------------

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._step += 1
        return self._finished

    def episode_callback(self):
        self._step = 0

    def render_callback(self, sim: Sim):
        now = self._step / self._hz
        draw_points(sim, self._traj(now).reshape(1, -1), rgba=(1, 0, 0, 1), size=0.02)
        draw_line(sim, self._traj(np.linspace(0, self._horizon, 100)), rgba=(0, 1, 0, 1))
