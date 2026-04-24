"""Simple adaptive racing controller for Levels 0-2.

Hardcoded gate-relative waypoints + CubicSpline trajectory.
PID with feedforward -> attitude commands.
RLS mass estimation for gravity compensation (Level 1).
Replans trajectory when sensor reveals actual gate positions (Level 2).

Track layout (nominal, from config):
  Gates:  [0.5,0.25,0.7]  [1.05,0.75,1.2]  [-1.0,-0.25,0.7]  [0.0,-0.75,1.2]
  Obstacles: [0.0,0.75] [1.0,0.25] [-1.5,-0.25] [-0.5,-0.75]
  Start: [-1.5, 0.75, 0.01] 
  Gate opening: 0.4m x 0.4m, outer frame 0.72m x 0.72m
  Obstacle poles: radius 0.015m, ~1.6m tall  
"""  

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class OnlineRLS:
    """Multi-parameter RLS estimator for drone dynamics parameters.

    Estimates theta = [m, J_xx, J_yy, J_zz] using separate RLS channels for
    translational (mass) and rotational (inertia) dynamics.
    """

    def __init__(
        self,
        nominal_mass: float,
        nominal_J: NDArray[np.floating],
        dt: float,
        lambda_mass: float = 0.995,
        lambda_inertia: float = 0.999,
    ):
        """Initialize the RLS estimator."""
        self._dt = dt
        self._g = 9.81
        self._nominal_mass = nominal_mass
        self._nominal_J_diag = np.array([nominal_J[0, 0], nominal_J[1, 1], nominal_J[2, 2]])
        self._n_params = 4
        self._theta = np.array(
            [
                nominal_mass,
                self._nominal_J_diag[0],
                self._nominal_J_diag[1],
                self._nominal_J_diag[2],
            ]
        )
        self._P = np.diag([1e2, 1e6, 1e6, 1e6])
        self._lambda_diag = np.array([lambda_mass, lambda_inertia, lambda_inertia, lambda_inertia])
        self._prev_vel: NDArray | None = None
        self._prev_ang_vel: NDArray | None = None
        self._prev_z_axis: NDArray | None = None
        self._prev_rpy_rates: NDArray | None = None
        self._n_updates = 0
        self._min_updates_for_valid = 10
        self._mass_bounds = (nominal_mass * 0.85, nominal_mass * 1.15)
        self._J_bounds = (self._nominal_J_diag * 0.5, self._nominal_J_diag * 2.0)

    @property
    def mass(self) -> float:
        """Current mass estimate."""
        return float(self._theta[0])

    @property
    def J_diag(self) -> NDArray[np.floating]:
        """Current inertia diagonal estimate."""
        return self._theta[1:4].copy()

    @property
    def covariance(self) -> NDArray[np.floating]:
        """Current parameter covariance matrix."""
        return self._P.copy()

    @property
    def is_converged(self) -> bool:
        """Whether enough updates have been collected."""
        return self._n_updates >= self._min_updates_for_valid

    def posterior(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the Gaussian posterior for scenario sampling."""
        return self._theta.copy(), self._P.copy()

    def reset(self):
        """Reset estimator to initial state."""
        self._theta = np.array(
            [
                self._nominal_mass,
                self._nominal_J_diag[0],
                self._nominal_J_diag[1],
                self._nominal_J_diag[2],
            ]
        )
        self._P = np.diag([1e2, 1e6, 1e6, 1e6])
        self._prev_vel = None
        self._prev_ang_vel = None
        self._prev_z_axis = None
        self._prev_rpy_rates = None
        self._n_updates = 0

    def update(
        self,
        obs: dict[str, NDArray[np.floating]],
        thrust_cmd: float,
        torque_cmd: NDArray[np.floating],
    ):
        """Update parameter estimates from new observation."""
        vel = np.asarray(obs["vel"], dtype=np.float64)
        ang_vel = np.asarray(obs["ang_vel"], dtype=np.float64)
        quat = np.asarray(obs["quat"], dtype=np.float64)
        rot_mat = Rot.from_quat(quat).as_matrix()
        z_axis = rot_mat[:, 2]

        if self._prev_vel is None:
            self._prev_vel = vel.copy()
            self._prev_ang_vel = ang_vel.copy()
            self._prev_z_axis = z_axis.copy()
            return

        lin_accel = (vel - self._prev_vel) / self._dt
        ang_accel = (ang_vel - self._prev_ang_vel) / self._dt
        self._update_mass(lin_accel, thrust_cmd)
        self._update_inertia(ang_accel, ang_vel)
        self._prev_vel = vel.copy()
        self._prev_ang_vel = ang_vel.copy()
        self._prev_z_axis = z_axis.copy()
        self._n_updates += 1

    def _update_mass(self, lin_accel: NDArray, thrust_cmd: float):
        g_vec = np.array([0.0, 0.0, self._g])
        rhs = lin_accel + g_vec
        y = thrust_cmd * self._prev_z_axis[2]
        phi = rhs[2]
        if abs(phi) < 0.5:
            return
        phi_full = np.zeros(self._n_params)
        phi_full[0] = phi
        self._rls_update(y, phi_full)

    def _update_inertia(self, ang_accel: NDArray, ang_vel: NDArray):
        wx, wy, wz = ang_vel
        ax, ay, az = ang_accel
        prev_wx, prev_wy, prev_wz = self._prev_ang_vel
        gyro_mag = abs(wy * wz) + abs(wz * wx) + abs(wx * wy)
        if gyro_mag < 0.1:
            return
        alpha_mag = np.linalg.norm(ang_accel)
        if alpha_mag < 0.5:
            return
        J_xx_est, J_yy_est, J_zz_est = self._theta[1], self._theta[2], self._theta[3]
        tau_x_est = J_xx_est * ax - (J_yy_est - J_zz_est) * wy * wz
        tau_y_est = J_yy_est * ay - (J_zz_est - J_xx_est) * wz * wx
        tau_z_est = J_zz_est * az - (J_xx_est - J_yy_est) * wx * wy

        phi_x = np.array([0.0, ax, -wy * wz, wy * wz])
        if np.linalg.norm(phi_x[1:]) > 0.1:
            self._rls_update(tau_x_est, phi_x)

        phi_y = np.array([0.0, wz * wx, ay, -wz * wx])
        if np.linalg.norm(phi_y[1:]) > 0.1:
            self._rls_update(tau_y_est, phi_y)

        phi_z = np.array([0.0, -wx * wy, wx * wy, az])
        if np.linalg.norm(phi_z[1:]) > 0.1:
            self._rls_update(tau_z_est, phi_z)

    def _rls_update(self, y: float, phi: NDArray):
        y_pred = phi @ self._theta
        innovation = y - y_pred
        P_phi = self._P @ phi
        denom = 1.0 + phi @ P_phi
        K = P_phi / denom
        self._theta += K * innovation
        W = np.diag(1.0 / np.sqrt(self._lambda_diag))
        P_updated = self._P - np.outer(K, phi @ self._P)
        self._P = W @ P_updated @ W
        self._P = 0.5 * (self._P + self._P.T)
        min_eig = np.min(np.diag(self._P))
        if min_eig < 1e-12:
            self._P += np.eye(self._n_params) * 1e-10
        self._clamp_parameters()

    def _clamp_parameters(self):
        self._theta[0] = np.clip(self._theta[0], *self._mass_bounds)
        for i in range(3):
            self._theta[i + 1] = np.clip(
                self._theta[i + 1], self._J_bounds[0][i], self._J_bounds[1][i]
            )


class SimpleRacingController(Controller):
    """Simple adaptive racing controller for Levels 0-2."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the racing controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        # Drone parameters
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._nominal_mass = float(params["mass"])
        self._g = 9.81

        # RLS mass estimation (Level 1: randomized inertia/mass)
        self._rls = OnlineRLS(
            nominal_mass=self._nominal_mass, nominal_J=np.array(params["J"]), dt=self._dt
        )

        # PID gains
        self._kp = np.array([0.6, 0.6, 1.5])
        self._ki = np.array([0.05, 0.05, 0.05])
        self._kd = np.array([0.45, 0.45, 0.5])
        self._i_error = np.zeros(3)
        self._i_limit = np.array([2.0, 2.0, 0.4])
        self._ff_gain = 0.3

        # Track geometry
        self._gate_positions = np.array(obs["gates_pos"], dtype=np.float64)
        self._gate_quats = np.array(obs["gates_quat"], dtype=np.float64)
        self._obstacle_positions = np.array(obs["obstacles_pos"], dtype=np.float64)
        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])
        self._gates_visited = np.array(obs["gates_visited"], dtype=bool)
        self._obstacles_visited = np.array(
            obs.get("obstacles_visited", np.zeros(len(self._obstacle_positions), dtype=bool)),
            dtype=bool,
        )

        # State
        self._tick = 0
        self._finished = False
        self._t_min = 0.0
        self._dip_done = False  # gate-2 dip sequence completed?

        # Build initial trajectory
        self._build_trajectory(np.array(obs["pos"], dtype=np.float64))

    # ---- Helpers ----

    def _gate_normal(self, quat: np.ndarray) -> np.ndarray:
        """Gate fly-through direction (local x-axis)."""
        return Rot.from_quat(quat).as_matrix()[:, 0]

    # ---- Waypoint generation ----

    def _compute_waypoints(self, start_pos: np.ndarray) -> np.ndarray:
        """Build waypoints from start_pos through remaining gates.

        Route modeled after attitude_controller's proven waypoints but using
        actual gate positions so it auto-adapts when gates shift (Level 2).
        Key routing: fly OUTWARD around obstacle poles, not between them.
        """
        g = self._gate_positions
        target = self._target_gate

        wps = [start_pos.copy()]

        # Takeoff
        if start_pos[2] < 0.1 and target == 0:
            wps.append(start_pos + np.array([0.5, -0.2, 0.4]))

        for i in range(target, self._n_gates):
            if i == 0:
                # Gate 0 normal=(+0.71,-0.70): approach from (-x,+y) side
                wps.append(g[0] + np.array([-0.2, 0.15, 0.0]))
                wps.append(g[0].copy())
            elif i == 1:
                # After gate 0: swing OUTWARD (south) around obstacle 1 [1.0,0.25]
                # Corner-rounding: two intermediate points to smooth the turn
                # so the interpolator doesn't overshoot at the sharp corner.
                wps.append(np.array([0.9, -0.05, 0.8]))  # ease into the turn
                wps.append(np.array([1.3, -0.15, 0.9]))  # apex south of obs1
                wps.append(np.array([1.3, 0.30, 1.05]))  # round the corner north
                wps.append(g[1].copy())
            elif i == 2:
                # Gate 2 normal=(-1.0,0): approach from +x side, exit -x
                wps.append(np.array([-0.5, -0.05, 0.7]))
                wps.append(g[2] + np.array([0.35, 0.08, 0.0]))
                wps.append(g[2].copy())
            elif i == 3:
                if not self._dip_done:
                    # "DIP" maneuver for gate 2 exit:
                    # Pole at [-1.5,-0.25] is right behind gate 2 [-1.0,-0.25,0.7].
                    # Enter gate, go barely past, come back out at SAME height.
                    # Climb only after safely back on the east (+x) side.
                    # 1. Barely past gate plane (level, same z)
                    wps.append(g[2] + np.array([-0.12, 0.0, 0.0]))
                    # 2. Ease back east (corner-rounding for the dip reversal)
                    # Must stay on exit side of gate (local_x > 0, world_x < gate_x)
                    wps.append(g[2] + np.array([-0.06, 0.0, 0.0]))  # still past gate
                    wps.append(g[2] + np.array([0.4, 0.0, 0.0]))  # fully east
                    # 3. Climb to gate 3 altitude right here (vertical ascent)
                    wps.append(g[2] + np.array([0.4, 0.0, g[3][2] - g[2][2]]))
                # 4. At altitude, head toward gate 3 approach
                # Route directly toward gate 3, staying >0.25m from obs3 [-0.5,-0.75].
                # Must be >0.3m from gate 3 plane in local_x to avoid gate frame
                # check clamping (gate 3 normal=+x, so need world_x < g3_x - 0.3).
                wps.append(np.array([-0.35, -0.45, g[3][2]]))
                wps.append(g[3].copy())
                wps.append(g[3] + np.array([0.5, 0.0, 0.0]))

        return np.array(wps)

    def _ensure_obstacle_clearance(self, wps: np.ndarray) -> np.ndarray:
        """Push waypoints away from obstacles and gate frames.

        Obstacles are treated as vertical cylinders (XY clearance).
        Gate frames are treated as keep-out zones: any waypoint that is near a
        gate but NOT passing through the opening gets pushed away.
        Drone radius ~0.05m (Crazyflie 92mm motor-to-motor) is accounted for.
        """
        obs_xy = self._obstacle_positions[:, :2]
        # Per-obstacle safe distances: obs 1 and 3 are on tight corners, use smaller margin
        safe_dists = np.array([0.25, 0.15, 0.25, 0.15])

        # Gate opening: 0.4m x 0.4m, outer frame 0.72m x 0.72m
        # With drone radius 0.05m, effective opening = 0.3m x 0.3m
        half_opening = 0.14  # conservative: 0.20 - 0.05 drone - 0.01 margin
        half_outer = 0.36  # outer frame half-width
        gate_safe = 0.15  # minimum clearance from frame edge

        # Pass 0: ensure waypoints near gates pass through the opening, not the frame
        for i in range(1, len(wps)):
            for gi in range(self._n_gates):
                gp = self._gate_positions[gi]
                R_gate = Rot.from_quat(self._gate_quats[gi]).as_matrix()
                local = R_gate.T @ (wps[i] - gp)
                # Only check points that are close to the gate plane (|local_x| < 0.3)
                if abs(local[0]) > 0.3:
                    continue
                # Check if in the frame zone but outside opening
                in_frame_y = abs(local[1]) < half_outer + gate_safe
                in_frame_z = abs(local[2]) < half_outer + gate_safe
                in_opening_y = abs(local[1]) < half_opening
                in_opening_z = abs(local[2]) < half_opening
                if in_frame_y and in_frame_z and not (in_opening_y and in_opening_z):
                    # Push through center of opening (clamp local_y and local_z)
                    local[1] = np.clip(local[1], -half_opening * 0.5, half_opening * 0.5)
                    local[2] = np.clip(local[2], -half_opening * 0.5, half_opening * 0.5)
                    wps[i] = gp + R_gate @ local

        # Pass 1: push existing waypoints away from obstacle poles (skip first = current pos)
        for i in range(1, len(wps)):
            for j, o in enumerate(obs_xy):
                sd = safe_dists[j] if j < len(safe_dists) else 0.25
                d = np.linalg.norm(wps[i, :2] - o)
                if 0 < d < sd:
                    wps[i, :2] = o + (wps[i, :2] - o) / d * sd

        # Pass 2: check line segments, insert avoidance waypoints where path is too close
        # Skip first segment (index 0→1) — drone is at wps[0], can't go backward
        new_wps = [wps[0]]
        for i in range(1, len(wps)):
            if i == 1:
                # First segment from current position — never insert avoidance
                new_wps.append(wps[i])
                continue

            a, b = wps[i - 1], wps[i]
            ab = b[:2] - a[:2]
            ab_len2 = float(ab.dot(ab))

            worst_idx = -1
            worst_margin = 0.0  # how far inside the safe zone (higher = worse)
            worst_t = 0.5

            if ab_len2 > 1e-12:
                for j, o in enumerate(obs_xy):
                    sd = safe_dists[j] if j < len(safe_dists) else 0.25
                    t = np.clip(float((o - a[:2]).dot(ab) / ab_len2), 0.0, 1.0)
                    closest = a[:2] + t * ab
                    d = float(np.linalg.norm(closest - o))
                    if d < sd and 0.05 < t < 0.95:
                        margin = sd - d
                        if margin > worst_margin:
                            worst_margin = margin
                            worst_idx = j
                            worst_t = t

            if worst_idx >= 0:
                o = obs_xy[worst_idx]
                sd = safe_dists[worst_idx] if worst_idx < len(safe_dists) else 0.25
                mid_3d = a + worst_t * (b - a)
                direction = mid_3d[:2] - o
                dn = float(np.linalg.norm(direction))
                direction = direction / dn if dn > 1e-6 else np.array([0.0, 1.0])
                avoid = mid_3d.copy()
                avoid[:2] = o + direction * (sd + 0.05)
                new_wps.append(avoid)

            new_wps.append(wps[i])

        return np.array(new_wps)

    # ---- Trajectory building ----

    def _build_trajectory(self, start_pos: np.ndarray):
        """Build CubicSpline trajectory through gate-relative waypoints."""
        if self._target_gate < 0:
            wp = np.array([start_pos, start_pos + [0, 0, 0.01]])
            self._spline = PchipInterpolator([0.0, 1.0], wp)
            self._vel_spline = self._spline.derivative()
            self._acc_spline = self._spline.derivative(2)
            self._t_total = 1.0
            self._t_offset = self._tick / self._freq
            self._gate_knot_mask = np.zeros(2, dtype=bool)
            self._wps = wp.copy()
            self._t_knots = np.array([0.0, 1.0])
            return

        wps = self._compute_waypoints(start_pos)
        wps = self._ensure_obstacle_clearance(wps)

        # Mark which knots are gate centers (must not be skipped)
        self._gate_knot_mask = np.zeros(len(wps), dtype=bool)
        for gi in range(self._n_gates):
            gp = self._gate_positions[gi]
            for wi in range(len(wps)):
                if np.linalg.norm(wps[wi] - gp) < 0.05:
                    self._gate_knot_mask[wi] = True

        # Time allocation: distance / cruise_speed, minimum 0.15s per segment
        # Corner penalty: slow down at sharp turns to reduce overshoot
        dists = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        cruise_speed = 1.0
        times = np.maximum(dists / cruise_speed, 0.15)
        # Compute direction changes and penalize sharp corners
        for si in range(1, len(dists)):
            d_prev = wps[si] - wps[si - 1]
            d_next = wps[si + 1] - wps[si]
            n_prev = np.linalg.norm(d_prev)
            n_next = np.linalg.norm(d_next)
            if n_prev > 1e-6 and n_next > 1e-6:
                cos_angle = np.clip(d_prev.dot(d_next) / (n_prev * n_next), -1, 1)
                # cos_angle=1 means straight, cos_angle=-1 means U-turn
                if cos_angle < 0.3:  # >72 degree turn
                    penalty = 1.0 + 0.7 * (1.0 - cos_angle)  # up to 2.4x for U-turn
                    times[si] *= penalty  # only slow the outgoing segment
        t_knots = np.concatenate([[0.0], np.cumsum(times)])

        self._spline = PchipInterpolator(t_knots, wps)
        self._vel_spline = self._spline.derivative()
        self._acc_spline = self._spline.derivative(2)
        self._t_total = float(t_knots[-1])
        self._t_offset = self._tick / self._freq
        self._t_min = 0.0  # monotonic time floor — never go backward

        # Store waypoints + knot times for local patching later
        self._wps = wps.copy()
        self._t_knots = t_knots.copy()

    def _patch_trajectory(self, shifted_obstacle_indices: list[int]):
        """Locally adjust waypoints near specific shifted obstacles only.

        Only touches future waypoints that are close to the obstacles that
        actually moved. Gate knots are never moved. Keeps the overall path
        shape and timing intact — no drastic mid-flight changes.
        """
        if not shifted_obstacle_indices:
            return
        obs_xy = self._obstacle_positions[:, :2]
        safe_dists = np.array([0.25, 0.15, 0.25, 0.15])
        wps = self._wps.copy()
        t_knots = self._t_knots.copy()
        changed = False

        # Current time — only patch future waypoints
        t_now = self._tick / self._freq - self._t_offset
        t_now = max(t_now, self._t_min)

        # Pass 1: push future soft waypoints away from SHIFTED obstacles only
        for wi in range(len(wps)):
            if t_knots[wi] < t_now - 0.1:
                continue  # already passed
            if self._gate_knot_mask[wi]:
                continue  # gate center — never move
            for oi in shifted_obstacle_indices:
                o = obs_xy[oi]
                sd = safe_dists[oi] if oi < len(safe_dists) else 0.25
                d = float(np.linalg.norm(wps[wi, :2] - o))
                if 0 < d < sd:
                    wps[wi, :2] = o + (wps[wi, :2] - o) / d * sd
                    changed = True

        # Pass 2: check future segments for proximity to SHIFTED obstacles
        new_wps = [wps[0]]
        new_mask = [self._gate_knot_mask[0]]
        new_t = [t_knots[0]]
        for i in range(1, len(wps)):
            if t_knots[i] > t_now and i > 1:
                a, b = wps[i - 1], wps[i]
                ab = b[:2] - a[:2]
                ab_len2 = float(ab.dot(ab))
                if ab_len2 > 1e-12:
                    for oi in shifted_obstacle_indices:
                        o = obs_xy[oi]
                        sd = safe_dists[oi] if oi < len(safe_dists) else 0.25
                        t_param = np.clip(float((o - a[:2]).dot(ab) / ab_len2), 0.0, 1.0)
                        closest = a[:2] + t_param * ab
                        d = float(np.linalg.norm(closest - o))
                        if d < sd and 0.1 < t_param < 0.9:
                            mid_3d = a + t_param * (b - a)
                            direction = mid_3d[:2] - o
                            dn = float(np.linalg.norm(direction))
                            if dn > 1e-6:
                                direction /= dn
                            else:
                                direction = np.array([0.0, 1.0])
                            avoid = mid_3d.copy()
                            avoid[:2] = o + direction * (sd + 0.05)
                            mid_t = (t_knots[i - 1] + t_knots[i]) * 0.5
                            new_wps.append(avoid)
                            new_mask.append(False)
                            new_t.append(mid_t)
                            changed = True
                            break  # one insertion per segment

            new_wps.append(wps[i])
            new_mask.append(self._gate_knot_mask[i] if i < len(self._gate_knot_mask) else False)
            new_t.append(t_knots[i])

        if changed:
            new_wps = np.array(new_wps)
            new_t = np.array(new_t)
            new_mask = np.array(new_mask, dtype=bool)
            self._spline = PchipInterpolator(new_t, new_wps)
            self._vel_spline = self._spline.derivative()
            self._acc_spline = self._spline.derivative(2)
            self._t_total = float(new_t[-1])
            self._wps = new_wps
            self._t_knots = new_t
            self._gate_knot_mask = new_mask

    # ---- Control ----

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """PID + feedforward -> attitude command [roll, pitch, yaw, thrust]."""
        t = self._tick / self._freq - self._t_offset
        t = float(np.clip(t, 0.0, self._t_total))
        if t >= self._t_total:
            self._finished = True

        # Enforce monotonic time — never track backward
        t = max(t, self._t_min)

        # Two-tier waypoint advancement:
        #   Gate knots: must get within 0.15m before advancing past them
        #   Soft knots: advance when drone has passed along travel direction
        # This prevents hard U-turns toward missed soft waypoints while
        # ensuring the drone actually flies through gates.
        knots = self._spline.x
        pos = obs["pos"]
        while True:
            ki = int(np.searchsorted(knots, t, side="right")) - 1
            ki = max(0, min(ki, len(knots) - 2))
            next_ki = ki + 1
            if next_ki >= len(knots):
                break

            wp_next = self._spline(knots[next_ki])
            dist_to_next = float(np.linalg.norm(pos - wp_next))

            if self._gate_knot_mask[next_ki]:
                # Gate knot: only advance once we're close enough
                if dist_to_next < 0.15:
                    t = float(knots[next_ki]) + 0.001
                else:
                    break
            else:
                # Soft knot: advance if passed along travel direction OR close
                if dist_to_next < 0.10:
                    t = float(knots[next_ki]) + 0.001
                    continue
                wp_prev = self._spline(knots[ki])
                travel = wp_next - wp_prev
                travel_len = float(np.linalg.norm(travel))
                if travel_len < 1e-6:
                    t = float(knots[next_ki]) + 0.001
                    continue
                progress = float((pos - wp_next).dot(travel)) / travel_len
                if progress > -0.05:
                    t = float(knots[next_ki]) + 0.001
                else:
                    break

        t = float(np.clip(t, 0.0, self._t_total))
        self._t_min = t  # ratchet forward

        des_pos = self._spline(t)
        des_vel = self._vel_spline(t)
        des_acc = self._acc_spline(t)

        # PID
        pos_err = des_pos - obs["pos"]
        # Clamp position error to prevent violent corrections
        err_mag = float(np.linalg.norm(pos_err))
        if err_mag > 0.3:
            pos_err = pos_err * (0.3 / err_mag)
        vel_err = des_vel - obs["vel"]
        self._i_error = np.clip(self._i_error + pos_err * self._dt, -self._i_limit, self._i_limit)

        mass = self._rls.mass
        thrust_vec = (
            self._kp * pos_err
            + self._ki * self._i_error
            + self._kd * vel_err
            + self._ff_gain * mass * des_acc
        )
        thrust_vec[2] += mass * self._g

        # Thrust vector -> attitude
        z_axis = Rot.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_scalar = float(thrust_vec.dot(z_axis))

        t_norm = np.linalg.norm(thrust_vec)
        z_des = np.array([0.0, 0.0, 1.0]) if t_norm < 1e-6 else thrust_vec / t_norm

        x_c = np.array([1.0, 0.0, 0.0])
        y_des = np.cross(z_des, x_c)
        yn = np.linalg.norm(y_des)
        y_des = np.array([0.0, 1.0, 0.0]) if yn < 1e-6 else y_des / yn
        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack([x_des, y_des, z_des])
        euler = Rot.from_matrix(R_des).as_euler("xyz")

        # Clamp roll/pitch to prevent extreme tilts that destabilize the drone
        max_tilt = 0.45  # ~25 degrees
        euler[0] = np.clip(euler[0], -max_tilt, max_tilt)
        euler[1] = np.clip(euler[1], -max_tilt, max_tilt)

        return np.array([euler[0], euler[1], euler[2], thrust_scalar], dtype=np.float32)

    # ---- Callbacks ----

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Post-step update: RLS, gate tracking, and trajectory patching."""
        self._tick += 1

        # RLS mass update
        if len(action) >= 4:
            self._rls.update(obs, action[3], action[:3])

        self._target_gate = int(obs["target_gate"])

        # Detect dip completion: target is gate 3 and drone is well above gate 2
        if self._target_gate == 3 and not self._dip_done:
            mid_z = (self._gate_positions[2][2] + self._gate_positions[3][2]) / 2
            if obs["pos"][2] > mid_z:
                self._dip_done = True

        # Check for gate/obstacle position updates (Level 2)
        gate_shifted = False
        shifted_obstacles = []
        for i in range(self._n_gates):
            if obs["gates_visited"][i] and not self._gates_visited[i]:
                shift = float(np.linalg.norm(self._gate_positions[i] - obs["gates_pos"][i]))
                if shift > 0.03:
                    gate_shifted = True
                self._gate_positions[i] = np.array(obs["gates_pos"][i], dtype=np.float64)
                self._gate_quats[i] = np.array(obs["gates_quat"][i], dtype=np.float64)

        for i in range(len(self._obstacle_positions)):
            if obs["obstacles_visited"][i] and not self._obstacles_visited[i]:
                shift = float(np.linalg.norm(self._obstacle_positions[i] - obs["obstacles_pos"][i]))
                if shift > 0.03:
                    shifted_obstacles.append(i)
                self._obstacle_positions[i] = np.array(obs["obstacles_pos"][i], dtype=np.float64)

        self._gates_visited = np.array(obs["gates_visited"], dtype=bool)
        self._obstacles_visited = np.array(obs["obstacles_visited"], dtype=bool)

        if self._target_gate >= 0:
            if gate_shifted:
                # Gate shifted — full rebuild (approach/exit waypoints all change)
                self._build_trajectory(np.array(obs["pos"], dtype=np.float64))
            elif shifted_obstacles:
                # Only obstacles shifted — local patch near those obstacles only
                self._patch_trajectory(shifted_obstacles)

        return self._finished

    def episode_callback(self):
        """Reset internal state after an episode."""
        self._tick = 0
        self._finished = False
        self._i_error[:] = 0
        self._t_min = 0.0
        self._dip_done = False
        self._rls.reset()

    def episode_reset(self):
        """Reset the controller for a new episode."""
        self.episode_callback()

    def render_callback(self, sim: Sim):
        """Draw the planned trajectory and current setpoint."""
        from crazyflow.sim.visualize import draw_line, draw_points

        t = self._tick / self._freq - self._t_offset
        t = float(np.clip(t, 0.0, self._t_total))
        sp = self._spline(t)
        draw_points(sim, sp.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        pts = [self._spline(float(ts)) for ts in np.linspace(0, self._t_total, 100)]
        draw_line(sim, np.array(pts), rgba=(0.0, 1.0, 0.0, 1.0))

    def reset(self):
        """Reset internal variables."""
        self.episode_callback()
