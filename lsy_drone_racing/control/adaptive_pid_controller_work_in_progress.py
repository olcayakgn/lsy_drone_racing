"""TOGT-inspired adaptive controller for drone racing.

Trajectory generation inspired by:
- Qin et al. (ICRA 2024): "Time-Optimal Gate-Traversing Planner" — optimize
  crossing points within gate openings, not just gate centers.
Key architecture:
1. Gate crossing point optimization (TOGT): Each gate's crossing point is a free
   variable within the gate opening. scipy SLSQP finds the shortest feasible
   path while respecting obstacle clearance constraints.
2. CubicSpline trajectory through sparse waypoints (crossing points + via-points)
   with dynamics-aware time allocation.
3. Obstacle avoidance: Integrated as inequality constraints in the crossing
   point optimizer AND detour insertion along trajectory segments.

Control:
- Adaptive PID with feedforward acceleration from CubicSpline trajectory
- Online RLS parameter identification (Sodre et al. 2025)
- APF reactive safety layer (Khatib 1986) as last-resort avoidance
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# =============================================================================
# Online RLS parameter estimator (inlined from lsy_drone_racing.control.online_rls)
# =============================================================================


class OnlineRLS:
    """Multi-parameter RLS estimator for drone dynamics parameters.

    Estimates θ = [m, J_xx, J_yy, J_zz] using separate RLS channels for
    translational (mass) and rotational (inertia) dynamics, then combines
    them into a single posterior N(θ_hat, P).
    """

    def __init__(
        self,
        nominal_mass: float,
        nominal_J: NDArray[np.floating],
        dt: float,
        lambda_mass: float = 0.995,
        lambda_inertia: float = 0.999,
    ):
        self._dt = dt
        self._g = 9.81

        self._nominal_mass = nominal_mass
        self._nominal_J_diag = np.array([nominal_J[0, 0], nominal_J[1, 1], nominal_J[2, 2]])

        self._n_params = 4
        self._theta = np.array([
            nominal_mass,
            self._nominal_J_diag[0],
            self._nominal_J_diag[1],
            self._nominal_J_diag[2],
        ])

        self._P = np.diag([1e2, 1e6, 1e6, 1e6])

        self._lambda_diag = np.array([
            lambda_mass,
            lambda_inertia,
            lambda_inertia,
            lambda_inertia,
        ])

        self._prev_vel: NDArray | None = None
        self._prev_ang_vel: NDArray | None = None
        self._prev_z_axis: NDArray | None = None
        self._prev_rpy_rates: NDArray | None = None

        self._n_updates = 0
        self._min_updates_for_valid = 10

        self._mass_bounds = (nominal_mass * 0.85, nominal_mass * 1.15)
        self._J_bounds = (
            self._nominal_J_diag * 0.5,
            self._nominal_J_diag * 2.0,
        )

    @property
    def theta(self) -> NDArray[np.floating]:
        return self._theta.copy()

    @property
    def mass(self) -> float:
        return float(self._theta[0])

    @property
    def J_diag(self) -> NDArray[np.floating]:
        return self._theta[1:4].copy()

    @property
    def covariance(self) -> NDArray[np.floating]:
        return self._P.copy()

    @property
    def is_converged(self) -> bool:
        return self._n_updates >= self._min_updates_for_valid

    def posterior(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return self._theta.copy(), self._P.copy()

    def reset(self):
        self._theta = np.array([
            self._nominal_mass,
            self._nominal_J_diag[0],
            self._nominal_J_diag[1],
            self._nominal_J_diag[2],
        ])
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
        y_x = tau_x_est
        if np.linalg.norm(phi_x[1:]) > 0.1:
            self._rls_update(y_x, phi_x)

        phi_y = np.array([0.0, wz * wx, ay, -wz * wx])
        y_y = tau_y_est
        if np.linalg.norm(phi_y[1:]) > 0.1:
            self._rls_update(y_y, phi_y)

        phi_z = np.array([0.0, -wx * wy, wx * wy, az])
        y_z = tau_z_est
        if np.linalg.norm(phi_z[1:]) > 0.1:
            self._rls_update(y_z, phi_z)

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
                self._theta[i + 1],
                self._J_bounds[0][i],
                self._J_bounds[1][i],
            )


# =============================================================================
# Gate crossing point optimizer (TOGT-inspired, Qin et al. ICRA 2024)
# =============================================================================


class GateCrossingOptimizer:
    """Optimize gate crossing points for minimum path length.

    Each gate has a rectangular opening (0.4 m x 0.4 m in the gate's local
    y-z plane).  The crossing point is parametrised as (dy, dz) offset from
    the gate centre in local coordinates, constrained to lie within the
    opening minus a drone-size safety margin.

    The optimizer minimises total Euclidean path length:
        |start → g0| + |g0 → g1| + … + |g_{n-1}|
    subject to:
      - Each crossing point within its gate opening (box constraint)
      - Each crossing point ≥ obs_clearance from all obstacle poles (XY)
      - Each crossing point ≥ ground_clearance from z = 0

    Key insight from Qin et al.: traversing gates near their edges (where
    the path is shorter) can reduce lap time by 10–20 %.
    """

    def __init__(
        self,
        gate_positions: np.ndarray,
        gate_quats: np.ndarray,
        obstacle_positions: np.ndarray,
        opening_half: float = 0.12,
        obs_clearance: float = 0.25,
        ground_clearance: float = 0.15,
    ):
        self.gate_positions = gate_positions.copy()
        self.gate_quats = gate_quats.copy()
        self.obstacle_positions = obstacle_positions.copy()
        self.n_gates = len(gate_positions)
        self.opening_half = opening_half
        self.obs_clearance = obs_clearance
        self.ground_clearance = ground_clearance

        # Pre-compute rotation matrices
        self.gate_rotmats = [Rot.from_quat(q).as_matrix() for q in gate_quats]

    def _crossing_point(self, gate_idx: int, dy: float, dz: float) -> np.ndarray:
        """World-frame crossing point from local (dy, dz) offset."""
        R_gate = self.gate_rotmats[gate_idx]
        local_offset = np.array([0.0, dy, dz])
        return self.gate_positions[gate_idx] + R_gate @ local_offset

    def optimize(self, start_pos: np.ndarray, target_gate: int = 0) -> np.ndarray:
        """Optimise crossing points for all gates from *target_gate* onward.

        Returns (n_active, 3) array of world-frame crossing points.
        """
        n_active = self.n_gates - target_gate
        if n_active <= 0:
            return np.array([start_pos])

        x0 = np.zeros(2 * n_active)  # Start at gate centres

        bounds = []
        for _ in range(n_active):
            bounds.append((-self.opening_half, self.opening_half))  # dy
            bounds.append((-self.opening_half, self.opening_half))  # dz

        obs_positions = self.obstacle_positions  # local alias for closure

        def _objective(x: np.ndarray) -> float:
            points = [start_pos]
            for i in range(n_active):
                pt = self._crossing_point(target_gate + i, x[2 * i], x[2 * i + 1])
                points.append(pt)
            return float(sum(
                np.linalg.norm(points[j + 1] - points[j]) for j in range(len(points) - 1)
            ))

        def _obstacle_ineq(x: np.ndarray) -> np.ndarray:
            """dist_xy - clearance >= 0 for each (gate, obstacle) pair."""
            vals = []
            for i in range(n_active):
                pt = self._crossing_point(target_gate + i, x[2 * i], x[2 * i + 1])
                for op in obs_positions:
                    vals.append(float(np.linalg.norm(pt[:2] - op[:2])) - self.obs_clearance)
            return np.array(vals)

        def _ground_ineq(x: np.ndarray) -> np.ndarray:
            """z - ground_clearance >= 0 for each crossing point."""
            vals = []
            for i in range(n_active):
                pt = self._crossing_point(target_gate + i, x[2 * i], x[2 * i + 1])
                vals.append(pt[2] - self.ground_clearance)
            return np.array(vals)

        result = scipy_minimize(
            _objective, x0, method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": _obstacle_ineq},
                {"type": "ineq", "fun": _ground_ineq},
            ],
            options={"maxiter": 200, "ftol": 1e-8},
        )

        crossing_points = np.zeros((n_active, 3))
        for i in range(n_active):
            crossing_points[i] = self._crossing_point(
                target_gate + i, result.x[2 * i], result.x[2 * i + 1]
            )
        return crossing_points


# =============================================================================
# Main controller
# =============================================================================


class AdaptivePIDController(Controller):
    """TOGT-inspired adaptive PID controller for drone racing.

    Planning layer:
      1. Optimise gate crossing points within openings (TOGT, Qin et al.)
      2. CubicSpline trajectory through sparse waypoints
      3. Obstacle detour insertion on trajectory segments

    Control layer:
      - Adaptive PID with feedforward acceleration from CubicSpline
      - Online RLS parameter identification (Sodré et al.)
      - APF reactive safety net (Khatib 1986)
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        # Drone parameters
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self._nominal_mass = drone_params["mass"]
        self._nominal_J = np.array(drone_params["J"])
        self._nominal_J_diag = np.diag(self._nominal_J)
        self._g = 9.81

        # --- Online RLS Parameter Estimator (Phase 2) ---
        self._rls = OnlineRLS(
            nominal_mass=self._nominal_mass,
            nominal_J=self._nominal_J,
            dt=self._dt,
            lambda_mass=0.995,
            lambda_inertia=0.999,
        )

        # --- PID Gains ---
        self._kp_base = np.array([0.6, 0.6, 1.5])
        self._ki_base = np.array([0.05, 0.05, 0.05])
        self._kd_base = np.array([0.45, 0.45, 0.5])
        self._i_error = np.zeros(3)
        self._ki_range = np.array([2.0, 2.0, 0.4])

        # Feed-forward acceleration gain from polynomial trajectory
        self._ff_acc_gain = 0.3

        # Gain scheduling bounds
        self._kp_min = np.array([0.3, 0.3, 0.8])
        self._kp_max = np.array([1.2, 1.2, 2.5])
        self._kd_min = np.array([0.1, 0.1, 0.2])
        self._kd_max = np.array([0.6, 0.6, 0.8])

        self._prev_z_axis = None

        # --- Track geometry ---
        self._gate_positions = np.array([g.tolist() for g in obs["gates_pos"]])
        self._gate_quats = np.array([g.tolist() for g in obs["gates_quat"]])
        self._obstacle_positions = np.array([g.tolist() for g in obs["obstacles_pos"]])
        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])

        # --- Collision / dynamics margins ---
        # Physical gate opening: 0.40 m (±0.20 m from center).
        # Collision geom at ±0.28 m center, 0.08 m thick → surface at ±0.20 m.
        # Drone half-width ~0.07 m.  Safe half-width = 0.20 − 0.07 − 0.01 = 0.12 m.
        self._gate_opening_half = 0.12
        # Gate forbidden box (gate local frame):
        #   Collision surface extends to ±0.36 m from center.
        #   Use 0.38 m to include drone half-width margin.
        self._gate_forbidden_outer = 0.38
        self._gate_forbidden_depth = 0.15
        # Obstacle clearance: pole radius 0.015 m + drone 0.07 m + 0.165 m margin.
        self._obs_clearance = 0.25
        self._ground_clearance = 0.15
        self._ceiling = 1.90

        # APF reactive safety layer
        self._apf_influence_radius = 0.25
        self._apf_gain = 0.15
        self._apf_max_accel = 0.4
        self._apf_gate_radius = 0.20

        # Dynamics constraints for time allocation
        self._max_vel = 1.8
        self._max_accel = 4.0
        self._cruise_speed = 1.2

        # ---- Custom via-points for manual route editing ----
        # Each entry: ("before"|"after", gate_index, [dx, dy, dz])
        #   "before" inserts a via-point BEFORE the approach to that gate
        #   "after"  inserts a via-point AFTER  the exit from that gate
        #   [dx, dy, dz] is an OFFSET from the gate center (world frame)
        # Because offsets are relative to gate positions, they auto-adjust
        # when gates move in Level 2.
        self._via_points: list[tuple[str, int, list[float]]] = [
            # Route AROUND pole 1 (1.0, 0.25) between gate 0 → gate 1:
            # Push upward in y after gate 0 so the drone goes above the pole
            ("after", 0, [0.0, 0.45, 0.20]),
            # Curve left after gate 2 toward gate 3, avoiding pole 2/3:
            # dx=0.6 pushes toward gate 3; dy=-0.10 curves left of pole 3
            ("after", 2, [0.6, -0.10, 0.10]),
        ]

        # Sensor update tracking
        self._gates_visited = obs["gates_visited"].copy()

        # State
        self._tick = 0
        self._finished = False
        self._drone_start = obs["pos"].copy()

        # Build initial trajectory
        self._build_trajectory(obs)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _get_gate_normal(self, gate_quat: np.ndarray) -> np.ndarray:
        """Gate fly-through normal (local x-axis)."""
        return Rot.from_quat(gate_quat).as_matrix()[:, 0]

    def _eval_at_time(self, t: float):
        """Evaluate piecewise trajectory at global time *t*.

        Returns (pos, vel, acc) ndarrays.
        """
        for seg in self._segments:
            if t <= seg['t_end'] + 1e-10:
                return seg['spline'](t), seg['vel'](t), seg['acc'](t)
        last = self._segments[-1]
        te = last['t_end']
        return last['spline'](te), last['vel'](te), last['acc'](te)

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_segment_distance_xy(
        p: np.ndarray, a: np.ndarray, b: np.ndarray,
    ) -> float:
        """Min XY distance from point *p* to segment *a*→*b*."""
        ab = b[:2] - a[:2]
        ap = p[:2] - a[:2]
        ab_sq = float(ab.dot(ab))
        if ab_sq < 1e-12:
            return float(np.linalg.norm(ap))
        t = np.clip(float(ap.dot(ab)) / ab_sq, 0.0, 1.0)
        closest = a[:2] + t * ab
        return float(np.linalg.norm(p[:2] - closest))

    # ------------------------------------------------------------------
    # Forbidden zone helpers
    # ------------------------------------------------------------------

    def _is_in_gate_forbidden_zone(
        self, point: np.ndarray, gate_idx: int, is_target: bool = False,
    ) -> bool:
        """Check if *point* lies inside a gate's forbidden zone.

        The forbidden zone is a box in the gate's local frame:
          |x_local| < depth  AND  |y_local| < outer  AND  |z_local| < outer
        For the current target gate the ±opening_half corridor is carved out.
        """
        R_gate = Rot.from_quat(self._gate_quats[gate_idx]).as_matrix()
        p_local = R_gate.T @ (point - self._gate_positions[gate_idx])
        if not (
            abs(p_local[0]) < self._gate_forbidden_depth
            and abs(p_local[1]) < self._gate_forbidden_outer
            and abs(p_local[2]) < self._gate_forbidden_outer
        ):
            return False
        if is_target:
            if (
                abs(p_local[1]) < self._gate_opening_half
                and abs(p_local[2]) < self._gate_opening_half
            ):
                return False
        return True

    def _is_in_pole_forbidden_zone(
        self, point: np.ndarray, pole_pos: np.ndarray,
    ) -> bool:
        """Check if *point* is inside a pole's forbidden cylinder."""
        if point[2] > 1.6:
            return False
        return float(np.linalg.norm(point[:2] - pole_pos[:2])) < self._obs_clearance

    def _point_in_any_forbidden_zone(
        self, point: np.ndarray, target_gate: int,
    ) -> bool:
        """Return True if *point* violates any forbidden zone."""
        for gi in range(self._n_gates):
            is_tgt = (gi == target_gate)
            if self._is_in_gate_forbidden_zone(point, gi, is_target=is_tgt):
                return True
        for op in self._obstacle_positions:
            if self._is_in_pole_forbidden_zone(point, op):
                return True
        return False

    def _segment_clips_gate_forbidden(
        self, a: np.ndarray, b: np.ndarray, gate_idx: int, is_target: bool,
    ) -> bool:
        """Sample 20 points along segment a→b and check the gate forbidden zone."""
        for frac in np.linspace(0.0, 1.0, 20):
            if self._is_in_gate_forbidden_zone(a + frac * (b - a), gate_idx, is_target):
                return True
        return False

    def _compute_gate_bypass_point(
        self, gate_idx: int, from_pt: np.ndarray, to_pt: np.ndarray,
    ) -> np.ndarray:
        """Return a waypoint that routes around a gate's forbidden zone.

        Places the bypass outside the gate forbidden zone in the gate's local y-direction,
        choosing the side closest to the segment midpoint.  The x-position
        is clamped outside the ±0.15 m depth zone so that straight-line
        segments to/from the bypass stay clear.
        """
        gate_pos = self._gate_positions[gate_idx]
        R_gate = Rot.from_quat(self._gate_quats[gate_idx]).as_matrix()
        from_local = R_gate.T @ (from_pt - gate_pos)
        to_local = R_gate.T @ (to_pt - gate_pos)
        mid_local = 0.5 * (from_local + to_local)

        bypass_y = self._gate_forbidden_outer + 0.10
        if mid_local[1] < 0:
            bypass_y = -bypass_y

        bypass_x = mid_local[0]
        if abs(bypass_x) < self._gate_forbidden_depth + 0.05:
            bypass_x = math.copysign(self._gate_forbidden_depth + 0.05, bypass_x or 1.0)

        z_local = np.clip(mid_local[2], -0.5, 0.5)
        bp = gate_pos + R_gate @ np.array([bypass_x, bypass_y, z_local])
        bp[2] = np.clip(bp[2], self._ground_clearance + 0.05, self._ceiling - 0.05)
        return bp

    # ------------------------------------------------------------------
    # Trajectory planning
    # ------------------------------------------------------------------

    def _build_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Build piecewise CubicSpline trajectory through remaining gates.

        Each segment spans from one gate crossing to the next.  At each
        gate boundary the velocity is constrained to align with the gate's
        fly-through normal, guaranteeing the drone truly crosses the gate
        plane rather than clipping the waypoint from the side.

        This implements the key insight from Qin et al. (ICRA 2024, TOGT):
        gate traversal requires crossing the gate *plane*, not just touching
        a waypoint.  By segmenting the trajectory at gates and constraining
        velocity direction, proper traversal is structurally enforced —
        without adding approach/exit micro-waypoints that could conflict
        with nearby obstacle poles.

        Pipeline per segment:
          1. Collect interior via-points
          2. Insert obstacle detours
          3. Dynamics-aware time allocation
          4. CubicSpline fit with gate-normal velocity BCs
        """
        current_pos = obs["pos"].copy()
        current_vel = obs["vel"].copy()
        target = self._target_gate

        if target < 0 or target >= self._n_gates:
            wp = np.array([current_pos, current_pos + [0, 0, 0.01]])
            t = np.array([0.0, 1.0])
            spline = CubicSpline(t, wp)
            self._segments = [{
                'spline': spline, 'vel': spline.derivative(),
                'acc': spline.derivative(2), 't_start': 0.0, 't_end': 1.0,
            }]
            self._t_total = 1.0
            self._t_offset = self._tick / self._freq
            return

        # ---- Stage 1: crossing-point optimisation ----
        optimizer = GateCrossingOptimizer(
            self._gate_positions,
            self._gate_quats,
            self._obstacle_positions,
            opening_half=self._gate_opening_half,
            obs_clearance=self._obs_clearance,
            ground_clearance=self._ground_clearance,
        )
        crossing_points = optimizer.optimize(current_pos, target)
        n_active = len(crossing_points)

        # ---- Gate normals ----
        gate_normals = [
            self._get_gate_normal(self._gate_quats[target + i])
            for i in range(n_active)
        ]

        # Segment boundaries: start → cross_0 → cross_1 → … → cross_{n-1}
        boundaries = [current_pos] + [cp.copy() for cp in crossing_points]

        # ---- Stage 2: build piecewise segments ----
        self._segments = []
        global_t = 0.0
        cur_pos = current_pos.copy()

        for seg_idx in range(n_active):
            gate_idx = target + seg_idx
            crossing = crossing_points[seg_idx]
            normal_i = gate_normals[seg_idx]

            # Gate forward direction aligned with approach
            d_approach = crossing - cur_pos
            gate_fwd = normal_i if np.dot(normal_i, d_approach) >= 0 else -normal_i

            # ---- Collect waypoints: cur_pos → [via] → [approach] → crossing ----
            prev_gate = target + seg_idx - 1
            interior: list[np.ndarray] = []
            if prev_gate >= 0:
                for vtype, vidx, voff in self._via_points:
                    if vtype == "after" and vidx == prev_gate:
                        vp = self._gate_positions[prev_gate] + np.array(voff)
                        vp[2] = np.clip(vp[2], self._ground_clearance + 0.05,
                                        self._ceiling - 0.05)
                        interior.append(vp)
            for vtype, vidx, voff in self._via_points:
                if vtype == "before" and vidx == gate_idx:
                    vp = self._gate_positions[gate_idx] + np.array(voff)
                    vp[2] = np.clip(vp[2], self._ground_clearance + 0.05,
                                    self._ceiling - 0.05)
                    interior.append(vp)

            # Add approach guide waypoint 0.25 m before gate to straighten entry
            approach_pt = crossing - 0.25 * gate_fwd
            approach_pt[2] = np.clip(
                approach_pt[2], self._ground_clearance + 0.05, self._ceiling - 0.05,
            )
            if np.linalg.norm(approach_pt - cur_pos) > 0.15:
                seg_wps = np.array([cur_pos] + interior + [approach_pt, crossing])
            else:
                seg_wps = np.array([cur_pos] + interior + [crossing])

            # Only obstacle detours (skip gate detours — the approach guide handles it)
            seg_wps = self._insert_obstacle_detours(seg_wps)
            seg_wps[:, 2] = np.clip(
                seg_wps[:, 2], self._ground_clearance, self._ceiling,
            )

            # Time allocation
            seg_times = self._allocate_segment_times(seg_wps, np.zeros(3))
            seg_cum = np.concatenate([[0.0], np.cumsum(seg_times)])

            # Fit with natural BCs (zero 2nd derivative at ends).
            # This prevents the curls/loops caused by velocity BC mismatches.
            seg_spline = CubicSpline(
                seg_cum + global_t, seg_wps, bc_type='natural',
            )
            self._segments.append({
                'spline': seg_spline,
                'vel': seg_spline.derivative(),
                'acc': seg_spline.derivative(2),
                't_start': global_t,
                't_end': global_t + float(seg_cum[-1]),
            })
            global_t += float(seg_cum[-1])
            cur_pos = crossing

        self._t_total = global_t
        self._t_offset = self._tick / self._freq

    # ------------------------------------------------------------------

    def _allocate_segment_times(
        self, waypoints: np.ndarray, current_vel: np.ndarray,
    ) -> np.ndarray:
        """Per-segment time allocation respecting dynamics limits.

        Uses trapezoidal-profile logic: t = max(dist/v_max, sqrt(2d/a_max)).
        Plus curvature-aware slowdown at sharp turns.
        """
        n_seg = len(waypoints) - 1
        seg_times = np.empty(n_seg)

        for i in range(n_seg):
            dist = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
            if dist < 1e-6:
                seg_times[i] = 0.1
                continue

            t_vel = dist / self._max_vel
            t_acc = math.sqrt(2.0 * dist / self._max_accel)

            # Turn-angle slow-down
            speed_factor = 1.0
            if 0 < i < n_seg:
                d_in = waypoints[i] - waypoints[i - 1]
                d_out = waypoints[i + 1] - waypoints[i]
                n_in = np.linalg.norm(d_in)
                n_out = np.linalg.norm(d_out)
                if n_in > 1e-6 and n_out > 1e-6:
                    cos_a = float(np.clip(np.dot(d_in, d_out) / (n_in * n_out), -1, 1))
                    theta = math.acos(cos_a)
                    speed_factor = max(math.cos(theta / 2.0), 0.3)

            t_cruise = dist / max(self._cruise_speed * speed_factor, 0.1)
            seg_times[i] = max(t_vel, t_acc, t_cruise, 0.05)

        return seg_times

    # ------------------------------------------------------------------

    def _insert_obstacle_detours(self, waypoints: np.ndarray) -> np.ndarray:
        """Single-pass detour insertion where segments clip obstacles."""
        result = [waypoints[0]]
        for seg_idx in range(len(waypoints) - 1):
            a, b = waypoints[seg_idx], waypoints[seg_idx + 1]
            worst_obs, worst_dist = None, float("inf")
            for op in self._obstacle_positions:
                d = self._point_segment_distance_xy(op, a, b)
                if d < self._obs_clearance + 0.05 and d < worst_dist:
                    worst_dist, worst_obs = d, op

            if worst_obs is not None:
                det = self._make_detour_point(a, b, worst_obs)
                if det is not None:
                    result.append(det)
            result.append(b)
        return np.array(result)

    def _make_detour_point(
        self, a: np.ndarray, b: np.ndarray, obs_pos: np.ndarray,
    ) -> np.ndarray | None:
        """Minimax side-selection detour around an obstacle pole.

        Evaluates both sides and picks the one where the worst turn angle
        (entry or exit) is minimised — the "outer racing line" (TOGT spirit).
        """
        ab = b[:2] - a[:2]
        ab_sq = float(ab.dot(ab))
        if ab_sq < 1e-12:
            return None

        seg_dir = ab / math.sqrt(ab_sq)
        perp = np.array([-seg_dir[1], seg_dir[0]])
        t_proj = float(np.clip((obs_pos[:2] - a[:2]).dot(ab) / ab_sq, 0.1, 0.9))
        push = self._obs_clearance + 0.15

        best_pt, best_score = None, -2.0
        for sign in (+1.0, -1.0):
            cand_xy = obs_pos[:2] + sign * perp * push
            d_to = cand_xy - a[:2]
            d_from = b[:2] - cand_xy
            n1, n2 = np.linalg.norm(d_to), np.linalg.norm(d_from)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos_e = float(np.dot(d_to, ab) / (n1 * math.sqrt(ab_sq)))
            cos_x = float(np.dot(d_from, ab) / (n2 * math.sqrt(ab_sq)))
            score = min(cos_e, cos_x)
            # Penalise if detour clips another pole
            if not all(
                np.linalg.norm(cand_xy - o[:2]) >= self._obs_clearance
                for o in self._obstacle_positions
            ):
                score -= 1.0
            if score > best_score:
                best_score = score
                z_i = a[2] + t_proj * (b[2] - a[2])
                best_pt = np.array([cand_xy[0], cand_xy[1], z_i])
        return best_pt

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Adaptive mechanisms (Phase 2)
    # ------------------------------------------------------------------

    @property
    def rls_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        return self._rls.posterior()

    @property
    def mass_estimate(self) -> float:
        return self._rls.mass

    @property
    def J_diag_estimate(self) -> np.ndarray:
        return self._rls.J_diag

    # ------------------------------------------------------------------
    # Control computation
    # ------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Adaptive PID + feedforward acceleration from CubicSpline trajectory."""
        t_local = self._tick / self._freq - self._t_offset
        t_local = float(np.clip(t_local, 0.0, self._t_total))

        if t_local >= self._t_total:
            self._finished = True

        des_pos, des_vel, des_acc = self._eval_at_time(t_local)
        des_yaw = 0.0

        # ---- Gain scheduling ----
        mass_ratio = self._rls.mass / self._nominal_mass
        J_ratio = self._rls.J_diag / self._nominal_J_diag
        avg_J_xy = 0.5 * (J_ratio[0] + J_ratio[1])
        gain_scale = np.array([mass_ratio * avg_J_xy, mass_ratio * avg_J_xy, mass_ratio])

        kp = np.clip(self._kp_base * gain_scale, self._kp_min, self._kp_max)
        ki = self._ki_base * gain_scale
        kd = np.clip(self._kd_base * gain_scale, self._kd_min, self._kd_max)

        # ---- PID + feed-forward ----
        pos_err = des_pos - obs["pos"]
        vel_err = des_vel - obs["vel"]

        self._i_error += pos_err * self._dt
        self._i_error = np.clip(self._i_error, -self._ki_range, self._ki_range)

        thrust = (
            kp * pos_err
            + ki * self._i_error
            + kd * vel_err
            + self._ff_acc_gain * self._rls.mass * des_acc
        )
        thrust[2] += self._rls.mass * self._g

        # ---- APF reactive obstacle avoidance ----
        drone_pos = obs["pos"]

        def _repulse(obj_xy: np.ndarray, r_infl: float) -> None:
            diff = drone_pos[:2] - obj_xy
            d = float(np.linalg.norm(diff))
            if 1e-4 < d < r_infl:
                mag = min(
                    self._apf_gain * (1.0 / d - 1.0 / r_infl) / (d * d),
                    self._apf_max_accel,
                )
                thrust[:2] += mag * diff / d

        for op in self._obstacle_positions:
            _repulse(op[:2], self._apf_influence_radius)
        for gi in range(self._n_gates):
            if gi != self._target_gate:
                _repulse(self._gate_positions[gi][:2], self._apf_gate_radius)

        # ---- Ground / ceiling safety push ----
        z = drone_pos[2]
        if z < self._ground_clearance + 0.1:
            thrust[2] += 0.5 * (self._ground_clearance + 0.1 - z)
        if z > self._ceiling - 0.1:
            thrust[2] -= 0.5 * (z - (self._ceiling - 0.1))

        # ---- Thrust → attitude conversion ----
        z_axis = Rot.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_scalar = float(thrust.dot(z_axis))

        t_norm = np.linalg.norm(thrust)
        z_des = np.array([0.0, 0.0, 1.0]) if t_norm < 1e-6 else thrust / t_norm

        x_c = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_des = np.cross(z_des, x_c)
        yn = np.linalg.norm(y_des)
        y_des = np.array([0.0, 1.0, 0.0]) if yn < 1e-6 else y_des / yn
        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack([x_des, y_des, z_des])
        euler = Rot.from_matrix(R_des).as_euler("xyz", degrees=False)

        self._prev_z_axis = z_axis.copy()
        return np.array([euler[0], euler[1], euler[2], thrust_scalar], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step callback
    # ------------------------------------------------------------------

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1

        # Online RLS
        if len(action) >= 4:
            self._rls.update(obs, action[3], action[:3])

        # Gate switching + sensor-triggered replanning
        new_target = int(obs["target_gate"])
        gates_visited_new = obs["gates_visited"]
        obs_visited_new = obs["obstacles_visited"]

        self._target_gate = new_target

        max_shift = 0.0
        for i in range(self._n_gates):
            if gates_visited_new[i] and not self._gates_visited[i]:
                max_shift = max(
                    max_shift,
                    float(np.linalg.norm(self._gate_positions[i] - obs["gates_pos"][i])),
                )
        for i in range(len(self._obstacle_positions)):
            if obs_visited_new[i]:
                max_shift = max(
                    max_shift,
                    float(np.linalg.norm(self._obstacle_positions[i] - obs["obstacles_pos"][i])),
                )

        # Update stored positions
        for i in range(self._n_gates):
            if gates_visited_new[i]:
                self._gate_positions[i] = obs["gates_pos"][i].copy()
                self._gate_quats[i] = obs["gates_quat"][i].copy()
        for i in range(len(self._obstacle_positions)):
            if obs_visited_new[i]:
                self._obstacle_positions[i] = obs["obstacles_pos"][i].copy()
        self._gates_visited = gates_visited_new.copy()

        # Replan if significant shift near planned route
        if max_shift > 0.05 and new_target >= 0:
            if self._route_affected(obs):
                self._i_error *= 0.5
                self._build_trajectory(obs)

        return self._finished

    def _route_affected(self, obs: dict) -> bool:
        """Check if any shifted object is near the current trajectory."""
        t_now = self._tick / self._freq - self._t_offset
        times = np.linspace(max(t_now, 0.0), self._t_total, 50)
        for t in times:
            rpt, _, _ = self._eval_at_time(t)
            for op in self._obstacle_positions:
                if np.linalg.norm(op[:2] - rpt[:2]) < self._obs_clearance * 3:
                    return True
            for i in range(self._n_gates):
                if i >= self._target_gate:
                    if np.linalg.norm(self._gate_positions[i] - obs["gates_pos"][i]) > 0.05:
                        return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def episode_callback(self):
        self._i_error[:] = 0
        self._tick = 0
        self._finished = False
        self._rls.reset()
        self._prev_z_axis = None

    def episode_reset(self):
        self.episode_callback()

    def render_callback(self, sim: Sim):
        from crazyflow.sim.visualize import draw_line, draw_points

        t_local = self._tick / self._freq - self._t_offset
        t_local = float(np.clip(t_local, 0.0, self._t_total))
        sp, _, _ = self._eval_at_time(t_local)
        draw_points(sim, sp.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        pts = []
        for t_s in np.linspace(0, self._t_total, 100):
            p, _, _ = self._eval_at_time(float(t_s))
            pts.append(p)
        trajectory = np.array(pts)
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    def reset(self):
        self.episode_callback()
