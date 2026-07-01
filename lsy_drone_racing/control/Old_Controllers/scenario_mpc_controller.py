"""Altitude-safe 50 Hz reactive-turn backup-route scenario/MPPI drone-racing controller.

This version combines the stochastic scenario MPC / MPPI planner with the
altitude-safe world-frame specific-thrust convention from the acados MPC
controller. It does not require acados or CasADi.

Control convention used everywhere in this file:
    u = [ax, ay, az] is the desired WORLD-FRAME specific thrust vector.
    Hover is u = [0, 0, 9.81].
    The simulated translational acceleration is [ax, ay, az - g].

The attitude conversion intentionally preserves the commanded vertical
specific-thrust component after roll/pitch clipping. This avoids the common
launch/turn climb caused by computing thrust as mass * norm(u) when the
requested horizontal acceleration cannot actually be realized because the
attitude command is saturated.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from drone_models.core import load_params
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_EPS = 1e-9


def _vertical_preserving_thrust(
    az_world: float,
    roll: float,
    pitch: float,
    mass: float,
    thrust_min: float,
    thrust_max: float,
) -> float:
    """Compute thrust so the vertical world component matches az_world.

    After roll/pitch clipping, mass * norm(u) is unsafe because the clipped
    attitude cannot realize the requested horizontal component but would still
    apply the larger total thrust. This function keeps the vertical component
    controlled by dividing by cos(roll) * cos(pitch).
    """
    c = math.cos(float(roll)) * math.cos(float(pitch))
    c = max(c, 0.25)

    az_world = max(float(az_world), 0.1)
    thrust = mass * az_world / c
    return float(np.clip(thrust, thrust_min, thrust_max))


def accel_to_attitude(
    a_cmd: np.ndarray,
    yaw_des: float,
    mass: float,
    max_tilt: float,
    thrust_min: float,
    thrust_max: float,
) -> np.ndarray:
    """Convert world-frame specific thrust vector to [roll, pitch, yaw, thrust].

    a_cmd is not inertial acceleration. It is the desired world-frame specific
    thrust vector. Hover is [0, 0, 9.81].

    Important altitude fix:
      1. Compute desired roll/pitch from a_cmd.
      2. Clip roll/pitch.
      3. Recompute thrust to preserve a_cmd[2].
    """
    a_vec = np.asarray(a_cmd, dtype=np.float64).copy()
    a_vec[2] = max(float(a_vec[2]), 0.1)

    a_mag = float(np.linalg.norm(a_vec))
    if a_mag < 0.5:
        return np.array([0.0, 0.0, yaw_des, mass * 9.81], dtype=np.float64)

    z_des = a_vec / a_mag

    cy, sy = np.cos(yaw_des), np.sin(yaw_des)
    zx_local = cy * z_des[0] + sy * z_des[1]
    zy_local = -sy * z_des[0] + cy * z_des[1]
    zz_local = z_des[2]

    roll = float(np.arctan2(-zy_local, zz_local))
    pitch = float(np.arctan2(zx_local, np.sqrt(zy_local * zy_local + zz_local * zz_local)))

    roll = float(np.clip(roll, -max_tilt, max_tilt))
    pitch = float(np.clip(pitch, -max_tilt, max_tilt))

    thrust = _vertical_preserving_thrust(
        az_world=float(a_vec[2]),
        roll=roll,
        pitch=pitch,
        mass=mass,
        thrust_min=thrust_min,
        thrust_max=thrust_max,
    )

    return np.array([roll, pitch, yaw_des, thrust], dtype=np.float64)


class PMMRacingController(Controller):
    """Sampling/scenario MPC for drone racing.

    This replaces the acados solver by a stochastic MPC / MPPI-CEM style planner:
      1. Keep a warm-start Gaussian distribution over future thrust-acceleration sequences.
      2. Sample many smooth candidate sequences.
      3. Roll all candidates forward with a double-integrator model.
      4. Score progress, reference tracking, poles, gate frame bars, funnel/opening safety,
         altitude, smoothness, and bad gate crossings.
      5. Refit the mean/covariance from exponentially weighted good samples.
      6. Apply the first input of the updated distribution, with a risk-aware fallback toward
         the best safe sample when close to poles or gate frames.

    State used internally: position and velocity from obs; control is world specific thrust
    acceleration [ax, ay, az], where hover is [0, 0, 9.81].
    """

    # ------------------------- geometry / task parameters -------------------------
    OBSTACLE_RADIUS = 0.25          # pole safety radius for drone center [m]
    OBSTACLE_BUFFER = 0.08          # soft buffer outside pole radius [m]

    GATE_HALF_OPENING = 0.16        # allowed center half-width at gate plane [m]
    GATE_OUTER_HALF = 0.28          # gate bar centerline half-width [m]
    GATE_POST_RADIUS = 0.12         # bar avoidance radius for drone center [m]
    GATE_FRAME_BUFFER = 0.070       # robust but not over-conservative frame buffer
    GATE_CLEARANCE = 0.045          # tightened opening clearance, still feasible
    GATE_PLANE_SLAB = 0.16          # hard outside-opening penalty near plane [m]

    FUNNEL_LENGTH = 0.70            # distance around gate plane where opening narrows [m]
    FUNNEL_OUTER_HALF = 0.34        # half-width at outer funnel end [m]
    APPROACH_DIST = 0.45
    EXIT_DIST = 0.35
    ALIGN_START_DIST = 1.20

    GROUND_CLEARANCE = 0.10
    CEILING = 1.80

    V_CRUISE = 1.65
    V_GATE = 1.15
    V_MAX = 2.80

    # Altitude/actuator consistency parameters.
    MAX_TILT_CMD = 0.55             # final attitude command saturation [rad]; slightly higher for reactive turns
    MAX_TILT_MPC = 0.50             # planner lateral acceleration consistency [rad]; matches sharper turn authority
    AZ_UP_MAX = 1.25                # max vertical specific thrust above hover [m/s^2]
    AZ_DOWN_MAX = 1.75              # max vertical specific thrust below hover [m/s^2]
    VZ_REF_MAX = 0.40               # vertical reference speed clamp [m/s]
    LAUNCH_HOLD_TIME = 0.25         # keep initial altitude reference fixed briefly [s]
    LAUNCH_BLEND_TIME = 0.70        # blend vertical command toward launch hold [s]
    Z_HOLD_KP = 2.8
    Z_HOLD_KD = 2.0

    # ------------------------------ sampling setup -------------------------------
    MPC_N = 28                      # 28 * 0.120 = 3.36 s horizon; finer dynamics for sharper turns
    MPC_DT = 0.120                  # finer route/scenario step; same compute class but more reactive than 0.15 s
    K_SAMPLES = 24                  # fresh samples around the selected backup-style route
    N_ELITES = 1                    # action uses the selected fresh robust scenario
    TEMPERATURE = 50.0              # lower = greedier; avoids hover-like averaging
    NOISE_RHO = 0.84                # temporal correlation of sampled inputs
    CMD_FILTER_ALPHA = 0.86         # more reactive: trust the new selected/feedback command more than previous command
    RISK_BLEND_MARGIN = 0.07        # below this margin, blend toward best safe sample
    PLANNER_HZ = 50.0               # expensive scenario planner rate; target controller/planner frequency
    RENDER_EVERY = 1                # draw route/scenario lines every render frame
    DRAW_FULL_DEBUG_GEOMETRY = False # route/scenario lines stay visible; heavy rings/funnels off by default
    EVALUATE_MEAN_ROLLOUT = False    # command is scenario-committed; skipping mean rollout saves time

    # Scenario selection / closed-loop tracking.
    # Use robust elite averages instead of a single best sample to reduce frame-to-frame jumps.
    # A sample is only considered robust if it keeps this much clearance to poles/frame/altitude.
    SAFE_SELECTION_MARGIN = 0.020   # absolute non-crash threshold
    SELECTION_MARGIN = 0.075        # robust true clearance for commandable scenarios near gates
    ROBUST_CROSS_RADIUS = 0.090     # prefer crossings very close to the opening center
    LOOSE_CROSS_RADIUS = 0.125      # loose fallback; still inside the tightened opening
    ELITE_BLEND_COUNT = 1           # no multi-modal averaging near gate; use best robust fresh scenario only
    ELITE_TEMPERATURE = 18.0        # only relevant if ELITE_BLEND_COUNT is raised again

    # Every physics callback adds a small feedback term toward the currently selected
    # reference route. This turns the cached sequence from pure open loop into a
    # receding-horizon route tracker, which prevents outward drift in turns.
    TRACK_LOOKAHEAD = 0.060
    TRACK_KP = np.array([3.20, 3.20, 1.80], dtype=np.float64)
    TRACK_KD = np.array([1.95, 1.95, 1.15], dtype=np.float64)
    TRACK_MAX_XY = 2.35
    TRACK_MAX_Z = 0.65
    DEBUG_EVERY_PLANS = 25

    # Projection-based route tracking for sharp turns. The cached scenario is
    # useful as feedforward, but the real drone has attitude lag and may drift
    # outward in corners. These gains track the selected path by nearest-point
    # projection instead of only by time, add lateral velocity damping, and slow
    # the along-path speed when a turn is coming.
    USE_PROJECTED_ROUTE_TRACKING = True
    PATH_LOOKAHEAD_DIST = 0.32
    PATH_LOOKAHEAD_SPEED_GAIN = 0.08
    PATH_LOOKAHEAD_MAX = 0.58
    CROSS_TRACK_KP = 5.20
    CROSS_TRACK_KD = 3.10
    ALONG_TRACK_KP = 0.80
    ALONG_TRACK_KD = 0.85
    TURN_BRAKE_GAIN = 1.35
    TURN_SPEED_MIN = 0.95
    TURN_SPEED_MAX = 1.60
    TURN_ANGLE_FOR_SLOWDOWN = 0.55
    CROSS_TRACK_SLOWDOWN_START = 0.12
    CROSS_TRACK_SLOWDOWN_FULL = 0.34

    # Fresh-scenario mode: every 50 Hz planning tick samples from the current
    # route anchors only. Old MPPI means/covariances and old selected routes do
    # not bias the next scenario set. This prevents stale outward routes and
    # stale near-frame samples from pulling the new decision into the gate frame.
    FORGET_OLD_SCENARIOS = True
    USE_PREVIOUS_ROUTE_CANDIDATE = False
    RESET_SIGMA_EACH_PLAN = True
    COMMAND_ONLY_NONCRASHING = True

    SIGMA_INIT = np.array([2.8, 2.8, 1.8], dtype=np.float64)
    SIGMA_MIN = np.array([0.35, 0.35, 0.25], dtype=np.float64)
    SIGMA_MAX = np.array([5.2, 5.2, 3.2], dtype=np.float64)

    # ------------------------- deterministic route layer -------------------------
    # The route layer prevents the sampler from averaging many weak/random futures
    # into a hover command. Each control call creates several candidate corridors
    # to the gate, selects the best ones, seeds MPPI around their PD controls,
    # applies one step, then replans from the new state.
    ROUTE_PRESELECT_K = 6            # backup-style route candidates fully rollout-ranked
    ROUTE_TOP_K = 3                  # route anchors used by the scenario sampler
    ROUTE_MEAN_BLEND = 1.00         # fresh planning: center samples on the new route, not old scenario memory
    ROUTE_COMMAND_BLEND = 0.24      # small bias toward actual selected route anchor
    ROUTE_SAMPLE_SPACING = 0.18
    ROUTE_OBS_LOOKAHEAD_MARGIN = 0.55
    ROUTE_SIDE_OFFSETS = (0.0, 0.28, -0.28, 0.48, -0.48, 0.70, -0.70)
    ROUTE_ENTRY_Y_OFFSETS = (0.0, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30)
    ROUTE_ENTRY_Z_OFFSETS = (0.0, 0.10, -0.10)

    # ------------------------------- cost weights --------------------------------
    W_REF_POS = 11.0
    W_REF_VEL = 3.0
    W_TERMINAL_REF = 40.0
    W_PROGRESS = 26.0
    W_GATE_DISTANCE_STAGE = 5.5
    W_GATE_DISTANCE_TERMINAL = 42.0
    W_GATE_CLOSING = 28.0
    W_NOT_CROSSED_REACHABLE = 1800.0
    W_CROSS_CENTER = 1900.0
    W_BAD_CROSS = 30000.0
    BONUS_GOOD_CROSS = 2100.0

    W_POLE_BUFFER = 2600.0
    W_POLE_COLLISION = 25000.0
    W_POLE_NEAR_EXP = 15.0

    W_FRAME_BUFFER = 6200.0
    W_FRAME_COLLISION = 56000.0
    W_FRAME_SLAB = 26000.0
    W_FUNNEL = 900.0

    W_ALTITUDE = 6000.0
    W_ALTITUDE_HARD = 40000.0
    W_SPEED_LIMIT = 18.0
    W_INPUT = 0.025
    W_DINPUT = 0.075
    W_LATERAL = 0.020

    # ------------------------------ reactive layer -------------------------------
    APF_INFLUENCE = 0.36
    APF_GAIN = 0.22
    APF_MAX = 0.65

    def __init__(self, obs: dict[str, "NDArray[np.floating]"], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._g = 9.81
        self._dt = 1.0 / float(config.env.freq)

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(drone_params["mass"])
        self._thrust_min = float(drone_params["thrust_min"]) * 4.0
        self._thrust_max = float(drone_params["thrust_max"]) * 4.0
        self._mass_estimate = self._mass

        thrust_acc_min = self._thrust_min / self._mass
        thrust_acc_max = self._thrust_max / self._mass

        # The planner samples world-frame specific thrust. Keep vertical thrust
        # close to hover so sampled trajectories cannot create launch/turn climbs.
        self._a_min_z = max(thrust_acc_min, self._g - self.AZ_DOWN_MAX)
        self._a_max_z = min(thrust_acc_max, self._g + self.AZ_UP_MAX)
        if self._a_max_z <= self._a_min_z + 1e-6:
            # Extremely defensive fallback for unusual vehicle parameters.
            self._a_min_z = thrust_acc_min
            self._a_max_z = thrust_acc_max

        # Make sampled lateral acceleration consistent with the final attitude
        # saturation. The attitude converter will preserve vertical thrust.
        self._a_max_xy = self._g * math.tan(self.MAX_TILT_MPC)
        self._hover_u = np.array([0.0, 0.0, self._g], dtype=np.float64)

        self._gate_positions = np.array([g.tolist() for g in obs["gates_pos"]], dtype=np.float64)
        self._gate_quats = np.array([g.tolist() for g in obs["gates_quat"]], dtype=np.float64)
        self._gate_rotmats = [Rot.from_quat(q).as_matrix() for q in self._gate_quats]
        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])

        self._obstacle_positions = np.array(
            [p.tolist() for p in obs["obstacles_pos"]], dtype=np.float64,
        )
        self._gates_visited = obs["gates_visited"].copy()
        self._obstacles_visited = np.array(
            obs.get("obstacles_visited", np.zeros(len(self._obstacle_positions), dtype=bool)),
            dtype=bool,
        )

        seed_value = getattr(config, "seed", None)
        seed = int(seed_value) if seed_value is not None else None
        self._rng = np.random.default_rng(seed)

        self._mean_u = np.tile(self._hover_u, (self.MPC_N, 1))
        horizon_ramp = np.linspace(0.85, 1.20, self.MPC_N)[:, None]
        self._base_sigma = horizon_ramp * self.SIGMA_INIT[None, :]
        self._sigma = self._base_sigma.copy()

        # Run the expensive route/scenario planner at the requested 50 Hz.
        # If the simulator calls compute_control faster than 50 Hz, the fast path
        # holds the cached selected command between planner updates. The cached
        # sequence is indexed by elapsed physical time, not by callback count,
        # because MPC_DT is much larger than a typical simulator step.
        self._planner_interval_steps = max(1, int(round(1.0 / max(self.PLANNER_HZ * self._dt, _EPS))))
        self._last_planner_tick = -10**9
        self._cached_plan_tick = -10**9
        self._cached_u_sequence = np.tile(self._hover_u, (self.MPC_N, 1))
        self._cached_u_index = 0
        self._cached_u0 = self._hover_u.copy()
        self._cached_ref_pos = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)
        self._cached_ref_vel = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)
        init_z = float(np.asarray(obs.get("pos", [0.0, 0.0, 0.5]), dtype=np.float64)[2])
        self._cached_ref_pos[:, 2] = init_z
        self._last_elite_count = 0

        self._prev_accel = self._hover_u.copy()
        self._prev_output = np.array([0.0, 0.0, 0.0, self._mass * self._g], dtype=np.float64)
        self._last_pos = np.zeros(3, dtype=np.float64)
        self._last_best_traj = None
        self._last_mean_traj = None
        self._active_route_points: np.ndarray | None = None
        self._route_candidates: list[np.ndarray] = []
        self._route_costs: list[float] = []
        self._last_route_cost = math.inf
        self._last_route_anchor_cost = math.inf
        self._last_route_anchor_margin = math.inf
        self._last_route_anchor_crossed = False
        self._last_route_anchor_traj = None
        self._last_selected_idx = -1
        self._last_best_cost = math.inf
        self._last_best_margin = math.inf
        self._launch_z: float | None = None
        self._tick = 0
        self._finished = False
        self._last_plan_ms = 0.0
        self._plan_ms_ema = 0.0
        self._plan_counter = 0

        print(
            f"[SCENARIO-MPC] ready: K={self.K_SAMPLES}, N={self.MPC_N}, "
            f"dt={self.MPC_DT:.3f}s, horizon={self.MPC_N * self.MPC_DT:.2f}s, "
            f"planner_hz={self.PLANNER_HZ:.1f}, interval={self._planner_interval_steps} steps, "
            f"a_xy_max={self._a_max_xy:.2f}, a_z=[{self._a_min_z:.2f},{self._a_max_z:.2f}]"
        )

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    def _get_gate_normal(self, gi: int, from_pos: np.ndarray) -> np.ndarray:
        """Gate normal pointing from from_pos toward the gate and through it."""
        if gi < 0 or gi >= self._n_gates:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        normal = self._gate_rotmats[gi][:, 0].copy()
        to_gate = self._gate_positions[gi] - from_pos
        if float(np.dot(to_gate, normal)) < 0.0:
            normal = -normal
        n = np.linalg.norm(normal)
        return normal / max(float(n), _EPS)

    def _apply_launch_altitude_ramp(self, ref_pos: np.ndarray, ref_vel: np.ndarray) -> None:
        """Ramp reference altitude away from the launch altitude gradually.

        This prevents the first few horizons from immediately pulling toward a
        higher future gate and causing the vehicle to climb at launch.
        """
        if self._launch_z is None:
            return

        elapsed = max(0.0, (self._tick - 1) * self._dt)
        for i in range(self.MPC_N + 1):
            t_abs = elapsed + i * self.MPC_DT
            ramp_time = max(0.0, t_abs - self.LAUNCH_HOLD_TIME)
            z_low = self._launch_z - self.VZ_REF_MAX * ramp_time
            z_high = self._launch_z + self.VZ_REF_MAX * ramp_time

            ref_pos[i, 2] = float(np.clip(ref_pos[i, 2], z_low, z_high))
            ref_pos[i, 2] = float(np.clip(ref_pos[i, 2], self.GROUND_CLEARANCE, self.CEILING))
            ref_vel[i, 2] = float(np.clip(ref_vel[i, 2], -self.VZ_REF_MAX, self.VZ_REF_MAX))

    @staticmethod
    def _world_to_gate_local(points: np.ndarray, gp: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Batch local coordinates. R maps gate-local to world."""
        return (points - gp) @ R

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        """Clip one control vector or a batch/sequence of vectors."""
        out = np.array(u, dtype=np.float64, copy=True)
        xy = out[..., :2]
        xy_norm = np.linalg.norm(xy, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self._a_max_xy / np.maximum(xy_norm, _EPS))
        out[..., :2] = xy * scale
        out[..., 2] = np.clip(out[..., 2], self._a_min_z, self._a_max_z)
        return out

    def _limit_commanded_accel(self, u: np.ndarray) -> np.ndarray:
        """Final safety limit before converting specific thrust to attitude."""
        return self._clip_u(u)

    def _launch_altitude_guard(self, u0: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Blend vertical command with launch-altitude hold during the first moments."""
        if self._launch_z is None:
            return u0

        elapsed = max(0.0, (self._tick - 1) * self._dt)
        if elapsed >= self.LAUNCH_BLEND_TIME:
            return u0

        z_err = self._launch_z - float(pos[2])
        vz_err = -float(vel[2])
        az_hold = self._g + self.Z_HOLD_KP * z_err + self.Z_HOLD_KD * vz_err
        az_hold = float(np.clip(az_hold, self._a_min_z, self._a_max_z))

        alpha = 1.0 - elapsed / max(self.LAUNCH_BLEND_TIME, _EPS)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        guarded = np.asarray(u0, dtype=np.float64).copy()
        guarded[2] = alpha * az_hold + (1.0 - alpha) * guarded[2]
        return guarded

    def _distance_to_gate_bars(self, local: np.ndarray) -> np.ndarray:
        """Distance from positions in gate-local coordinates to the nearest frame bar centerline."""
        x = local[:, 0]
        y = local[:, 1]
        z = local[:, 2]
        oh = self.GATE_OUTER_HALF

        y_clamped = np.clip(y, -oh, oh)
        z_clamped = np.clip(z, -oh, oh)

        d_top_sq = x * x + (y - y_clamped) ** 2 + (z - oh) ** 2
        d_bottom_sq = x * x + (y - y_clamped) ** 2 + (z + oh) ** 2
        d_left_sq = x * x + (y + oh) ** 2 + (z - z_clamped) ** 2
        d_right_sq = x * x + (y - oh) ** 2 + (z - z_clamped) ** 2

        d_sq = np.minimum(np.minimum(d_top_sq, d_bottom_sq), np.minimum(d_left_sq, d_right_sq))
        return np.sqrt(np.maximum(d_sq, 0.0))

    def _gate_geometry_cost_and_margin(
        self,
        pos: np.ndarray,
        gp: np.ndarray,
        R: np.ndarray,
        active_funnel: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cost and safety margin for a batch of positions around one gate."""
        local = self._world_to_gate_local(pos, gp, R)
        x_abs = np.abs(local[:, 0])
        y_abs = np.abs(local[:, 1])
        z_abs = np.abs(local[:, 2])

        cost = np.zeros(pos.shape[0], dtype=np.float64)

        # Bar/corner frame as four 3-D capsule-like line segments.
        d_bar = self._distance_to_gate_bars(local)
        frame_soft = self.GATE_POST_RADIUS + self.GATE_FRAME_BUFFER
        v_soft = np.maximum(frame_soft - d_bar, 0.0)
        v_hard = np.maximum(self.GATE_POST_RADIUS - d_bar, 0.0)
        cost += self.W_FRAME_BUFFER * (v_soft / max(frame_soft, _EPS)) ** 2
        cost += self.W_FRAME_COLLISION * (v_hard / max(self.GATE_POST_RADIUS, _EPS)) ** 2

        # Near the gate plane, crossing outside the inner opening is very bad.
        in_slab = x_abs < self.GATE_PLANE_SLAB
        outside_open = np.maximum(y_abs - self.GATE_HALF_OPENING, z_abs - self.GATE_HALF_OPENING)
        outside_open = np.maximum(outside_open, 0.0)
        slab_scale = (1.0 - x_abs / self.GATE_PLANE_SLAB)
        slab_scale = np.maximum(slab_scale, 0.0)
        cost += self.W_FRAME_SLAB * in_slab * slab_scale * (outside_open / self.GATE_HALF_OPENING) ** 2

        # Funnel: away from the plane the allowed y/z corridor widens; at the plane it is the opening.
        if active_funnel:
            alpha = np.clip(x_abs / self.FUNNEL_LENGTH, 0.0, 1.0)
            h_bound = self.GATE_HALF_OPENING + alpha * (self.FUNNEL_OUTER_HALF - self.GATE_HALF_OPENING)
            v_y = np.maximum(y_abs - h_bound, 0.0)
            v_z = np.maximum(z_abs - h_bound, 0.0)
            funnel_weight = 1.0 - 0.55 * alpha
            cost += self.W_FUNNEL * funnel_weight * ((v_y / h_bound) ** 2 + (v_z / h_bound) ** 2)

        margin_bar = d_bar - self.GATE_POST_RADIUS
        return cost, margin_bar

    # -------------------------------------------------------------------------
    # Deterministic route candidates
    # -------------------------------------------------------------------------
    def _clip_route_point(self, point: np.ndarray) -> np.ndarray:
        """Return a copy of point with altitude inside the flyable volume."""
        p = np.asarray(point, dtype=np.float64).copy()
        p[2] = float(np.clip(p[2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05))
        return p

    def _clean_route(self, points: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Remove duplicate waypoints and clip waypoint altitude."""
        arr = [self._clip_route_point(np.asarray(q, dtype=np.float64)) for q in points]
        if not arr:
            return np.zeros((0, 3), dtype=np.float64)

        cleaned = [arr[0]]
        for q in arr[1:]:
            if float(np.linalg.norm(q - cleaned[-1])) > 0.06:
                cleaned.append(q)
        if len(cleaned) == 1:
            cleaned.append(cleaned[0].copy())
        return np.vstack(cleaned)

    @staticmethod
    def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > _EPS:
            return np.asarray(v, dtype=np.float64) / n
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).copy()

    @staticmethod
    def _point_segment_distance_xy(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Distance from point to segment in xy, plus segment parameter and closest xy."""
        ab = b_xy - a_xy
        denom = float(np.dot(ab, ab))
        if denom < _EPS:
            return float(np.linalg.norm(point_xy - a_xy)), 0.0, a_xy.copy()
        t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
        closest = a_xy + t * ab
        return float(np.linalg.norm(point_xy - closest)), t, closest

    def _sample_polyline(self, route: np.ndarray, spacing: float | None = None) -> np.ndarray:
        """Sample points along a route for cheap geometric scoring."""
        if spacing is None:
            spacing = self.ROUTE_SAMPLE_SPACING
        route = np.asarray(route, dtype=np.float64)
        if len(route) <= 1:
            return route.copy()

        pts: list[np.ndarray] = [route[0].copy()]
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            d = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(d / max(spacing, 0.03))))
            for j in range(1, n + 1):
                alpha = j / n
                pts.append((1.0 - alpha) * a + alpha * b)
        return np.vstack(pts)

    def _make_gate_tail(self, pos: np.ndarray) -> list[np.ndarray]:
        """Waypoints that force the route through current gate and then toward next gate."""
        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            return [pos.copy()]

        gp = self._gate_positions[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)
        approach = gp - self.APPROACH_DIST * normal
        entry = gp - self.FUNNEL_LENGTH * normal
        exit_pt = gp + self.EXIT_DIST * normal

        signed = float(np.dot(pos - gp, normal))
        tail: list[np.ndarray] = []

        # If we are still clearly before the plane, enter the funnel from a
        # proper pre-gate waypoint. If we are already close, do not backtrack.
        if signed < -0.20 and float(np.linalg.norm(pos - entry)) > 0.20:
            tail.append(entry)
        if signed < -0.04 and float(np.linalg.norm(pos - approach)) > 0.16:
            tail.append(approach)

        tail.append(gp.copy())
        tail.append(exit_pt)

        next_gate = self._target_gate + 1
        if next_gate < self._n_gates:
            gp_next = self._gate_positions[next_gate]
            normal_next = self._get_gate_normal(next_gate, exit_pt)
            tail.append(gp_next - self.APPROACH_DIST * normal_next)
            tail.append(gp_next.copy())

        return [self._clip_route_point(q) for q in tail]

    def _score_route(self, route: np.ndarray, pos: np.ndarray) -> float:
        """Cheap deterministic score for route candidates before MPPI rollout."""
        route = self._clean_route(route)
        if len(route) <= 1:
            return 1e9

        diffs = route[1:] - route[:-1]
        seg_lengths = np.linalg.norm(diffs, axis=1)
        length = float(np.sum(seg_lengths))
        score = 1.00 * length

        # Prefer an initial command that actually points generally toward the target gate.
        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            first_dir = self._unit(route[1] - route[0])
            gate_dir = self._unit(gp - pos, fallback=first_dir)
            align = float(np.dot(first_dir, gate_dir))
            score += 7.0 * max(0.0, 1.0 - align)

            # If no pole forces a detour, the first waypoint should stay close
            # to the straight line toward the gate. This keeps the displayed
            # and followed best route intuitively aimed at gate 0. A blocked
            # straight path still loses because obstacle penalties are much larger.
            cross_track, _, _ = self._point_segment_distance_xy(route[1, :2], pos[:2], gp[:2])
            score += 26.0 * cross_track * cross_track

        # Penalize sharp turns; route candidates should be flyable by the low-level controller.
        for i in range(1, len(route) - 1):
            a = self._unit(route[i] - route[i - 1])
            b = self._unit(route[i + 1] - route[i])
            score += 0.45 * max(0.0, 1.0 - float(np.dot(a, b))) ** 2

        sampled = self._sample_polyline(route, spacing=self.ROUTE_SAMPLE_SPACING)

        # Cylindrical pole clearance in xy. These penalties are intentionally high so
        # a slightly longer route around a pole beats a short route through it.
        if self._obstacle_positions.size > 0:
            soft = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER + 0.12
            for op in self._obstacle_positions:
                d = np.linalg.norm(sampled[:, :2] - op[:2], axis=1)
                d_min = float(np.min(d))
                if d_min < soft:
                    score += 180.0 * ((soft - d_min) / soft) ** 2
                if d_min < self.OBSTACLE_RADIUS:
                    score += 1800.0 * ((self.OBSTACLE_RADIUS - d_min) / self.OBSTACLE_RADIUS) ** 2

        # Keep route scoring cheap. All generated routes are explicitly forced
        # through the gate center, so detailed gate-frame/funnel costs are handled
        # by the rollout scorer instead of repeated here for every candidate.
        # This removes many small-array numpy calls and is important for fluid 50 Hz use.

        # Altitude should remain comfortable; launch ramp later handles early vertical motion.
        z = sampled[:, 2]
        below = np.maximum(self.GROUND_CLEARANCE + 0.04 - z, 0.0)
        above = np.maximum(z - (self.CEILING - 0.06), 0.0)
        score += 300.0 * float(np.mean(below * below + above * above))

        # Reward immediate reduction of gate distance. Without this, a geometrically
        # safe but indecisive sideways route can beat the route that actually starts
        # toward the gate, especially when all routes eventually pass through the
        # opening. This term is deliberately small compared with collision costs.
        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            d0 = float(np.linalg.norm(pos - gp))
            d1 = float(np.linalg.norm(route[min(1, len(route) - 1)] - gp))
            score -= 12.0 * max(0.0, d0 - d1)

        return float(score)

    def _build_route_candidates(self, pos: np.ndarray, vel: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Build and rank deterministic waypoint routes through the active gate.

        This is the important commitment layer. MPPI is not asked to discover the
        whole topology from random controls. Instead it receives several plausible
        corridors: direct, funnel-entry offsets, global doglegs, and pole-specific
        detours. The best route is then followed for one control step and the
        process repeats at the next step.
        """
        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            route = self._clean_route([pos.copy(), pos.copy()])
            return [route], [0.0]

        gp = self._gate_positions[self._target_gate]
        R = self._gate_rotmats[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)
        gate_y = R[:, 1].copy()
        gate_z = R[:, 2].copy()
        if float(np.linalg.norm(gate_y)) < _EPS:
            gate_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if float(np.linalg.norm(gate_z)) < _EPS:
            gate_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        gate_y = self._unit(gate_y)
        gate_z = self._unit(gate_z)

        entry_center = gp - self.FUNNEL_LENGTH * normal
        approach = gp - self.APPROACH_DIST * normal
        exit_pt = gp + self.EXIT_DIST * normal
        tail = self._make_gate_tail(pos)

        candidates: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()

        def add_route(points: list[np.ndarray]) -> None:
            route = self._clean_route(points)
            if len(route) < 2:
                return
            key = tuple(np.round(route.reshape(-1), 2).tolist())
            if key in seen:
                return
            seen.add(key)
            candidates.append(route)

        # 1) Must-have direct routes. These make the no-obstacle best route point
        # exactly toward the gate instead of drifting sideways.
        add_route([pos.copy(), gp.copy(), exit_pt] + tail[3:])
        add_route([pos.copy()] + tail)
        add_route([pos.copy(), approach, gp.copy(), exit_pt] + tail[3:])

        # 1b) Previous selected route as a continuity candidate. Without this, a
        # small outward drift can make the next replan choose an even more outward
        # dogleg, causing wide turns. Reusing the remaining old route gives the
        # scorer a stable centerline to keep following when it is still safe.
        prev_route = self._active_route_points
        if self.USE_PREVIOUS_ROUTE_CANDIDATE and prev_route is not None and len(prev_route) > 2:
            prev = np.asarray(prev_route, dtype=np.float64)
            idx_prev = int(np.argmin(np.linalg.norm(prev - pos[None, :], axis=1)))
            if idx_prev < len(prev) - 1:
                add_route([pos.copy()] + [q.copy() for q in prev[idx_prev + 1:]])

        # 2) Funnel entry offsets. These are still forced through the center of
        # the opening, but allow a smooth approach from different sides/heights.
        for dy in self.ROUTE_ENTRY_Y_OFFSETS:
            for dz in self.ROUTE_ENTRY_Z_OFFSETS:
                if abs(dy) < _EPS and abs(dz) < _EPS:
                    continue
                entry = entry_center + dy * gate_y + dz * gate_z
                app = approach + 0.45 * dy * gate_y + 0.45 * dz * gate_z
                add_route([pos.copy(), entry, app, gp.copy(), exit_pt] + tail[3:])

        # 3) Global doglegs around the direct line to the funnel/gate. Sideways
        # routes are longer, so they only win when poles/frame costs justify them.
        to_entry = entry_center - pos
        side = np.array([-to_entry[1], to_entry[0], 0.0], dtype=np.float64)
        if float(np.linalg.norm(side[:2])) < _EPS:
            side = gate_y.copy()
        side = self._unit(side)
        mid = pos + 0.45 * to_entry
        for off in self.ROUTE_SIDE_OFFSETS:
            if abs(off) < _EPS:
                continue
            dogleg = mid + off * side
            dogleg[2] = 0.65 * pos[2] + 0.35 * entry_center[2]
            add_route([pos.copy(), dogleg, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        # 4) Pole-specific detours. If an obstacle lies close to the current
        # straight corridor, create two route options around it.
        corridor_target = entry_center if float(np.linalg.norm(entry_center - pos)) > 0.25 else gp
        a_xy = pos[:2]
        b_xy = corridor_target[:2]
        seg_xy = b_xy - a_xy
        side_xy = np.array([-seg_xy[1], seg_xy[0]], dtype=np.float64)
        side_norm = float(np.linalg.norm(side_xy))
        if side_norm < _EPS:
            side_xy = gate_y[:2]
            side_norm = max(float(np.linalg.norm(side_xy)), _EPS)
        side_xy = side_xy / side_norm

        for op in self._obstacle_positions:
            d_seg, t_seg, closest_xy = self._point_segment_distance_xy(op[:2], a_xy, b_xy)
            if 0.02 < t_seg < 0.98 and d_seg < self.OBSTACLE_RADIUS + self.ROUTE_OBS_LOOKAHEAD_MARGIN:
                z_detour = (1.0 - t_seg) * pos[2] + t_seg * entry_center[2]
                for sign in (-1.0, 1.0):
                    for extra in (0.36,):
                        clearance = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER + extra
                        detour_xy = op[:2] + sign * clearance * side_xy
                        detour = np.array([detour_xy[0], detour_xy[1], z_detour], dtype=np.float64)
                        add_route([pos.copy(), detour, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        # 5) A velocity-biased route can prevent the optimizer from asking for an
        # immediate sharp reversal if the drone already has useful momentum.
        speed = float(np.linalg.norm(vel[:2]))
        if speed > 0.25:
            v_dir = self._unit(np.array([vel[0], vel[1], 0.0], dtype=np.float64), fallback=side)
            look = pos + min(0.55, 0.25 * speed) * v_dir
            look[2] = pos[2]
            add_route([pos.copy(), look, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        if not candidates:
            add_route([pos.copy()] + tail)

        scored = [(self._score_route(route, pos), route) for route in candidates]
        scored.sort(key=lambda item: item[0])
        # Keep more candidates than we finally sample. The cheap geometric score can
        # be wrong because it does not know whether the PD rollout can actually
        # pass the gate. A second, full rollout ranking is done in compute_control.
        top = scored[:max(1, int(self.ROUTE_PRESELECT_K))]
        routes = [r for _, r in top]
        scores = [float(c) for c, _ in top]
        return routes, scores

    # -------------------------------------------------------------------------
    # Reference and deterministic samples
    # -------------------------------------------------------------------------
    def _generate_references(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        route_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate references by marching along a selected deterministic route."""
        ref_pos = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)
        ref_vel = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)

        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            ref_pos[:] = pos
            return ref_pos, ref_vel

        if route_points is None:
            routes, _ = self._build_route_candidates(pos, vel)
            route = routes[0]
        else:
            route = self._clean_route(route_points)

        # Always make the first point exactly the current position. The route is
        # rebuilt every step, so stale initial waypoints must not pull backward.
        route = route.copy()
        route[0] = pos.copy()
        route[0, 2] = float(np.clip(route[0, 2], self.GROUND_CLEARANCE, self.CEILING))

        if len(route) < 2:
            ref_pos[:] = pos
            self._apply_launch_altitude_ramp(ref_pos, ref_vel)
            return ref_pos, ref_vel

        seg_vecs = route[1:] - route[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(cum[-1])
        if total < _EPS:
            ref_pos[:] = pos
            self._apply_launch_altitude_ramp(ref_pos, ref_vel)
            return ref_pos, ref_vel

        gp = self._gate_positions[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)

        speed0 = max(float(np.linalg.norm(vel)), 0.15)
        # Do not let a stationary start make the reference crawl. The controller
        # needs a committed moving reference immediately.
        avg_speed = min(self.V_CRUISE, max(0.85 * self.V_CRUISE, 0.5 * (speed0 + self.V_CRUISE)))

        for k in range(self.MPC_N + 1):
            t = k * self.MPC_DT
            s = min(avg_speed * t, total)

            seg = int(np.searchsorted(cum, s, side="right") - 1)
            seg = int(np.clip(seg, 0, len(seg_lens) - 1))
            seg_len = max(float(seg_lens[seg]), _EPS)
            alpha = float(np.clip((s - cum[seg]) / seg_len, 0.0, 1.0))
            ref_pos[k] = (1.0 - alpha) * route[seg] + alpha * route[seg + 1]
            ref_pos[k, 2] = float(np.clip(ref_pos[k, 2], self.GROUND_CLEARANCE, self.CEILING))

            if k < self.MPC_N:
                tangent = self._unit(route[seg + 1] - route[seg], fallback=normal)
                dist_to_gate = float(np.linalg.norm(ref_pos[k] - gp))

                # Inside the alignment zone, blend from route tangent to gate normal.
                # This keeps obstacle detours far away from the gate, but forces a
                # clean through-the-center direction near the frame.
                if dist_to_gate < self.ALIGN_START_DIST:
                    blend = 1.0 - dist_to_gate / max(self.ALIGN_START_DIST, _EPS)
                    blend = float(np.clip(blend, 0.0, 1.0))
                    tangent = self._unit((1.0 - blend) * tangent + blend * normal, fallback=normal)
                    v_des = self.V_GATE + (self.V_CRUISE - self.V_GATE) * (1.0 - blend)
                else:
                    v_des = self.V_CRUISE

                ref_vel[k] = tangent * v_des
                ref_vel[k, 2] = float(np.clip(ref_vel[k, 2], -self.VZ_REF_MAX, self.VZ_REF_MAX))

        ref_vel[self.MPC_N] = ref_vel[self.MPC_N - 1]
        ref_vel[self.MPC_N, 2] = float(np.clip(ref_vel[self.MPC_N, 2], -self.VZ_REF_MAX, self.VZ_REF_MAX))

        self._apply_launch_altitude_ramp(ref_pos, ref_vel)
        return ref_pos, ref_vel

    def _make_pd_sequence(self, pos: np.ndarray, vel: np.ndarray,
                          ref_pos: np.ndarray, ref_vel: np.ndarray) -> np.ndarray:
        """A deterministic stabilizing sequence inserted among the random scenarios."""
        seq = np.zeros((self.MPC_N, 3), dtype=np.float64)
        p = pos.astype(np.float64).copy()
        v = vel.astype(np.float64).copy()
        # XY gains are deliberately assertive: this PD sequence is not the final
        # low-level controller, it is a route-following anchor for MPPI. The final
        # command is still clipped by the tilt-consistent acceleration limits.
        kp = np.array([4.7, 4.7, 4.6], dtype=np.float64)
        kd = np.array([2.8, 2.8, 2.5], dtype=np.float64)

        for k in range(self.MPC_N):
            e_p = ref_pos[k + 1] - p
            e_v = ref_vel[k + 1] - v
            u = self._hover_u + kp * e_p + kd * e_v
            u = self._clip_u(u)
            seq[k] = u

            net_a = u - self._hover_u
            p = p + v * self.MPC_DT + 0.5 * net_a * self.MPC_DT * self.MPC_DT
            v = v + net_a * self.MPC_DT

        return seq

    def _shift_distribution(self) -> None:
        """Warm-start the next control step from the previous solution."""
        self._mean_u[:-1] = self._mean_u[1:]
        self._mean_u[-1] = self._mean_u[-2]
        self._mean_u = self._clip_u(self._mean_u)

        self._sigma[:-1] = self._sigma[1:]
        self._sigma[-1] = np.maximum(self._sigma[-2], self.SIGMA_INIT)
        self._sigma = np.clip(self._sigma, self.SIGMA_MIN, self.SIGMA_MAX)

    def _rank_routes_by_rollout(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        routes: list[np.ndarray],
        route_costs: list[float],
    ) -> tuple[list[np.ndarray], list[float], list[np.ndarray], np.ndarray, np.ndarray]:
        """Rank the few preselected route candidates by a full deterministic rollout.

        This is intentionally the same route-decision idea from the uploaded good backup,
        but only ROUTE_PRESELECT_K candidates are evaluated and MPC_N is smaller,
        so it stays inside the 50 Hz budget.

        """
        if not routes:
            ref_pos, ref_vel = self._generate_references(pos, vel, None)
            seq = self._make_pd_sequence(pos, vel, ref_pos, ref_vel)
            return [], [], [seq], ref_pos, ref_vel

        records = []
        for idx, route in enumerate(routes):
            r_pos, r_vel = self._generate_references(pos, vel, route)
            seq = self._make_pd_sequence(pos, vel, r_pos, r_vel)
            c, margin, traj, crossed, bad = self._score_rollouts(
                seq[None, :, :], pos, vel, r_pos, r_vel,
            )

            c0 = float(c[0])
            m0 = float(margin[0])
            crossed0 = bool(crossed[0])
            bad0 = bool(bad[0])
            geom = float(route_costs[idx]) if idx < len(route_costs) else 0.0

            # Selection score: actual rollout dominates. Geometry only breaks ties.
            # A clean crossing wins over a safe non-crossing route; unsafe routes are
            # kept as fallback but should almost never be selected.
            select_score = c0 + 0.04 * geom
            if crossed0 and not bad0 and m0 > -0.01:
                select_score -= 4500.0
            if not crossed0:
                select_score += 850.0
            if bad0:
                select_score += 20000.0
            if m0 < 0.0:
                select_score += 12000.0 * (0.02 - m0) ** 2 + 2500.0

            safety_bucket = 0 if (crossed0 and not bad0 and m0 > 0.0) else (1 if m0 > 0.0 else 2)
            records.append((
                safety_bucket, select_score, idx, route, geom, seq, r_pos, r_vel,
                c0, m0, crossed0, bad0, traj[0].astype(np.float64),
            ))

        records.sort(key=lambda item: (item[0], item[1]))
        keep = records[:max(1, int(self.ROUTE_TOP_K))]

        routes_ranked = [rec[3] for rec in keep]
        costs_ranked = [float(rec[4]) for rec in keep]
        seqs_ranked = [rec[5] for rec in keep]
        best = keep[0]

        self._last_route_cost = float(best[4])
        self._last_route_anchor_cost = float(best[8])
        self._last_route_anchor_margin = float(best[9])
        self._last_route_anchor_crossed = bool(best[10])
        self._last_route_anchor_traj = best[12]

        return routes_ranked, costs_ranked, seqs_ranked, best[6], best[7]

    def _sample_sequences(self, route_seqs: list[np.ndarray]) -> np.ndarray:
        """Create fresh smooth scenario samples around current route anchors only.

        This robust-fresh version deliberately does not use the previous MPPI mean
        as a sampling center. A stale mean can keep old side-drift/frame-grazing
        behavior alive even after the newest geometry says the center route is
        better. Each 50 Hz replan therefore samples only around the current route
        anchors generated from the current state.
        """
        K = int(self.K_SAMPLES)
        N = int(self.MPC_N)

        cleaned: list[np.ndarray] = []
        for seq in route_seqs:
            arr = self._clip_u(np.asarray(seq, dtype=np.float64))
            if arr.shape == (N, 3):
                cleaned.append(arr)
        if not cleaned:
            cleaned = [np.tile(self._hover_u, (N, 1))]

        # Reset exploration scale each replan when requested. This keeps only the
        # current-state route anchors, not old scenario statistics.
        if self.FORGET_OLD_SCENARIOS:
            sigma = self._base_sigma.copy()
            self._sigma = sigma.copy()
        else:
            sigma = self._sigma

        n_det = min(max(5, 2 * len(cleaned) + 3), max(1, K // 2))
        n_rand = max(K - n_det, 1)
        half = max(n_rand // 2, 1)

        eps_half = self._rng.standard_normal((half, N, 3)) * sigma[None, :, :]
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        while eps.shape[0] < n_rand:
            extra = self._rng.standard_normal((1, N, 3)) * sigma[None, :, :]
            eps = np.concatenate([eps, extra], axis=0)
        eps = eps[:n_rand]

        rho = float(self.NOISE_RHO)
        c = math.sqrt(max(1.0 - rho * rho, 0.0))
        smooth = np.empty_like(eps)
        smooth[:, 0, :] = eps[:, 0, :]
        for k in range(1, N):
            smooth[:, k, :] = rho * smooth[:, k - 1, :] + c * eps[:, k, :]

        # Current route anchors only. No previous mean center.
        center_bank = cleaned[:max(1, min(len(cleaned), self.ROUTE_TOP_K))]
        center_arr = np.stack(center_bank, axis=0)
        probs = np.exp(-0.70 * np.arange(len(center_bank), dtype=np.float64))
        probs /= max(float(np.sum(probs)), _EPS)
        center_idx = self._rng.choice(len(center_bank), size=n_rand, p=probs)

        samples = np.empty((K, N, 3), dtype=np.float64)
        samples[:n_rand] = center_arr[center_idx] + smooth

        # Deterministic anchors are exact current-route controllers and harmless
        # within-route variants. Do not include self._mean_u; it may represent an
        # old frame-grazing mode.
        deterministic: list[np.ndarray] = [cleaned[0].copy()]
        for seq in cleaned[1:]:
            deterministic.append(seq.copy())
            deterministic.append(0.70 * cleaned[0] + 0.30 * seq)

        # Small vertical variants for the best current route.
        up = cleaned[0].copy()
        down = cleaned[0].copy()
        up[:, 2] += 0.35
        down[:, 2] -= 0.35
        deterministic.extend([up, down])

        write = n_rand
        det_i = 0
        while write < K:
            samples[write] = deterministic[det_i % len(deterministic)]
            write += 1
            det_i += 1

        return self._clip_u(samples)

    # -------------------------------------------------------------------------
    # Rollout scoring
    # -------------------------------------------------------------------------
    def _score_rollouts(
        self,
        samples: np.ndarray,
        pos0: np.ndarray,
        vel0: np.ndarray,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Roll out all samples and return cost, min safety margin, best trajectory data."""
        K, N, _ = samples.shape
        pos = np.repeat(pos0[None, :], K, axis=0)
        vel = np.repeat(vel0[None, :], K, axis=0)
        last_u = np.repeat(self._prev_accel[None, :], K, axis=0)

        cost = np.zeros(K, dtype=np.float64)
        min_margin = np.full(K, np.inf, dtype=np.float64)
        best_stage_positions = np.empty((K, N + 1, 3), dtype=np.float32)
        best_stage_positions[:, 0, :] = pos.astype(np.float32)

        target_valid = 0 <= self._target_gate < self._n_gates
        if target_valid:
            gp = self._gate_positions[self._target_gate]
            R = self._gate_rotmats[self._target_gate]
            normal = self._get_gate_normal(self._target_gate, pos0)
            signed_prev = (pos - gp) @ normal
            signed0 = float((pos0 - gp) @ normal)
            max_progress = signed_prev.copy()
            good_crossed = np.zeros(K, dtype=bool)
            bad_crossed = np.zeros(K, dtype=bool)
            dist0_gate = float(np.linalg.norm(pos0 - gp))
            reachable_this_horizon = dist0_gate < max(0.85, 1.10 * self.V_MAX * self.MPC_N * self.MPC_DT)
        else:
            gp = np.zeros(3, dtype=np.float64)
            R = np.eye(3)
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            signed_prev = np.zeros(K, dtype=np.float64)
            signed0 = 0.0
            max_progress = np.zeros(K, dtype=np.float64)
            good_crossed = np.zeros(K, dtype=bool)
            bad_crossed = np.zeros(K, dtype=bool)
            reachable_this_horizon = False

        next_valid = target_valid and (self._target_gate + 1 < self._n_gates)
        if next_valid:
            gp_next = self._gate_positions[self._target_gate + 1]
            R_next = self._gate_rotmats[self._target_gate + 1]
        else:
            gp_next = None
            R_next = None

        # Only unvisited or current obstacles are considered. If the environment does not use
        # obstacles_visited, this still keeps all obstacles active.
        obstacle_positions = self._obstacle_positions

        for k in range(N):
            u = samples[:, k, :]
            du = u - last_u
            last_u = u

            # Control regularization. Obstacle/gate terms dominate these weights.
            cost += self.W_INPUT * np.sum((u - self._hover_u) ** 2, axis=1)
            cost += self.W_DINPUT * np.sum(du ** 2, axis=1)
            cost += self.W_LATERAL * np.sum(u[:, :2] ** 2, axis=1)

            pos_prev = pos.copy()
            net_a = u - self._hover_u
            pos = pos + vel * self.MPC_DT + 0.5 * net_a * self.MPC_DT * self.MPC_DT
            vel = vel + net_a * self.MPC_DT
            best_stage_positions[:, k + 1, :] = pos.astype(np.float32)

            stage = (k + 1) / N
            stage_w = 0.65 + 0.70 * stage

            # Soft reference tracking. This is intentionally weaker than obstacle/frame costs.
            e_p = pos - ref_pos[k + 1]
            e_v = vel - ref_vel[k + 1]
            cost += stage_w * (self.W_REF_POS * np.sum(e_p ** 2, axis=1)
                               + self.W_REF_VEL * np.sum(e_v ** 2, axis=1))

            # Altitude safety.
            below = np.maximum(self.GROUND_CLEARANCE - pos[:, 2], 0.0)
            above = np.maximum(pos[:, 2] - self.CEILING, 0.0)
            cost += self.W_ALTITUDE * (below ** 2 + above ** 2)
            hard_alt = (pos[:, 2] < 0.02) | (pos[:, 2] > self.CEILING + 0.25)
            cost += self.W_ALTITUDE_HARD * hard_alt
            min_margin = np.minimum(min_margin, pos[:, 2] - self.GROUND_CLEARANCE)
            min_margin = np.minimum(min_margin, self.CEILING - pos[:, 2])

            # Speed limit.
            speed = np.linalg.norm(vel, axis=1)
            overspeed = np.maximum(speed - self.V_MAX, 0.0)
            cost += self.W_SPEED_LIMIT * overspeed ** 2

            # Poles / cylindrical obstacles in xy.
            if obstacle_positions.size > 0:
                for op in obstacle_positions:
                    diff_xy = pos[:, :2] - op[:2]
                    d = np.sqrt(np.maximum(np.sum(diff_xy ** 2, axis=1), _EPS))
                    soft_r = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER
                    v_soft = np.maximum(soft_r - d, 0.0)
                    v_hard = np.maximum(self.OBSTACLE_RADIUS - d, 0.0)
                    cost += self.W_POLE_BUFFER * (v_soft / soft_r) ** 2
                    cost += self.W_POLE_COLLISION * (v_hard / self.OBSTACLE_RADIUS) ** 2
                    cost += self.W_POLE_NEAR_EXP * np.exp(-np.maximum(d - self.OBSTACLE_RADIUS, 0.0) / 0.18)
                    min_margin = np.minimum(min_margin, d - self.OBSTACLE_RADIUS)

            # Current target gate frame/funnel. The funnel is only active
            # before a trajectory has cleanly crossed the opening. After crossing,
            # the current gate remains a passive frame obstacle but no longer acts
            # as a corridor. This prevents the controller from rejecting the good
            # straight-through trajectory because it exits the old funnel.
            if target_valid:
                signed_now = (pos - gp) @ normal
                max_progress = np.maximum(max_progress, signed_now)

                still_approaching = ~(good_crossed | (signed_now > self.EXIT_DIST))
                if np.any(still_approaching):
                    c_gate, m_gate = self._gate_geometry_cost_and_margin(
                        pos[still_approaching], gp, R, active_funnel=True,
                    )
                    cost[still_approaching] += c_gate
                    min_margin[still_approaching] = np.minimum(min_margin[still_approaching], m_gate)
                if np.any(~still_approaching):
                    c_gate_passive, m_gate_passive = self._gate_geometry_cost_and_margin(
                        pos[~still_approaching], gp, R, active_funnel=False,
                    )
                    cost[~still_approaching] += 0.55 * c_gate_passive
                    min_margin[~still_approaching] = np.minimum(
                        min_margin[~still_approaching], m_gate_passive,
                    )

                # Strong gate-attraction before crossing. This fixes the common
                # sampling failure where many mediocre futures average to a
                # floating command instead of committing toward gate 0. Once a
                # sample crosses correctly, this term is disabled so it can exit
                # the gate and aim for the next one.
                dist_gate_now = np.linalg.norm(pos - gp, axis=1)
                dist_gate_prev = np.linalg.norm(pos_prev - gp, axis=1)
                not_crossed_yet = ~good_crossed
                norm_gate = max(dist0_gate, 1.0)
                cost += not_crossed_yet * self.W_GATE_DISTANCE_STAGE * stage_w * (dist_gate_now / norm_gate) ** 2
                cost -= not_crossed_yet * self.W_GATE_CLOSING * ((dist_gate_prev - dist_gate_now) / norm_gate)

                crossing = (~good_crossed) & (~bad_crossed) & (signed_prev <= 0.0) & (signed_now >= 0.0)
                if np.any(crossing):
                    denom = np.maximum(signed_now - signed_prev, _EPS)
                    alpha = np.clip(-signed_prev / denom, 0.0, 1.0)
                    p_cross = pos_prev + alpha[:, None] * (pos - pos_prev)
                    local_cross = self._world_to_gate_local(p_cross, gp, R)
                    y = local_cross[:, 1]
                    z = local_cross[:, 2]
                    err = y * y + z * z
                    clear = max(self.GATE_HALF_OPENING - self.GATE_CLEARANCE, 0.04)
                    inside = (np.abs(y) <= clear) & (np.abs(z) <= clear)
                    cross_center_cost = self.W_CROSS_CENTER * err
                    bad_amount = (np.maximum(np.abs(y) - clear, 0.0) ** 2
                                  + np.maximum(np.abs(z) - clear, 0.0) ** 2)

                    cost += crossing * cross_center_cost
                    cost += crossing * inside * (-self.BONUS_GOOD_CROSS)
                    cost += crossing * (~inside) * (self.W_BAD_CROSS * (1.0 + bad_amount / (clear * clear)))
                    good_crossed |= crossing & inside
                    bad_crossed |= crossing & (~inside)

                signed_prev = signed_now

            # Next gate frame is an obstacle too. After the sampled trajectory is
            # clearly through the current gate, activate the next gate funnel as well.
            if next_valid and gp_next is not None and R_next is not None:
                use_next_funnel = good_crossed | (signed_now > self.EXIT_DIST)

                if np.any(use_next_funnel):
                    c_next_active, m_next_active = self._gate_geometry_cost_and_margin(
                        pos[use_next_funnel], gp_next, R_next, active_funnel=True,
                    )
                    cost[use_next_funnel] += 0.75 * c_next_active
                    min_margin[use_next_funnel] = np.minimum(min_margin[use_next_funnel], m_next_active)

                if np.any(~use_next_funnel):
                    c_next_passive, m_next_passive = self._gate_geometry_cost_and_margin(
                        pos[~use_next_funnel], gp_next, R_next, active_funnel=False,
                    )
                    cost[~use_next_funnel] += 0.55 * c_next_passive
                    min_margin[~use_next_funnel] = np.minimum(min_margin[~use_next_funnel], m_next_passive)

        # Terminal cost and progress reward.
        terminal_error = pos - ref_pos[-1]
        cost += self.W_TERMINAL_REF * np.sum(terminal_error ** 2, axis=1)

        if target_valid:
            local_final = self._world_to_gate_local(pos, gp, R)
            lateral_final = local_final[:, 1] ** 2 + local_final[:, 2] ** 2
            signed_final = (pos - gp) @ normal

            # Reward progress along the correct gate normal but also penalize
            # terminal distance to the gate before crossing. Signed-normal progress
            # alone is not enough when the drone starts laterally offset from the
            # gate; distance reduction makes the best sample point at gate 0.
            cost -= self.W_PROGRESS * (max_progress - signed0)
            dist_final_to_gate = np.linalg.norm(pos - gp, axis=1)
            miss_terminal = ~good_crossed
            norm_gate = max(dist0_gate, 1.0)
            cost += miss_terminal * self.W_GATE_DISTANCE_TERMINAL * (dist_final_to_gate / norm_gate) ** 2
            cost -= good_crossed * 250.0

            near_plane = np.exp(-np.abs(local_final[:, 0]) / 0.45)
            cost += 180.0 * near_plane * lateral_final

            if reachable_this_horizon:
                # If the gate should be reachable inside the horizon, missing the opening is expensive.
                miss = ~good_crossed
                cost += self.W_NOT_CROSSED_REACHABLE * miss * np.maximum(0.15 - signed_final, 0.0) ** 2
                cost += 450.0 * miss * np.maximum(-signed_final, 0.0) ** 2

            if next_valid and gp_next is not None:
                # Once through the gate, prefer being on the way to the next approach point.
                normal_next = self._get_gate_normal(self._target_gate + 1, gp + normal * self.EXIT_DIST)
                next_approach = gp_next - self.APPROACH_DIST * normal_next
                d_next = np.sum((pos - next_approach) ** 2, axis=1)
                cost += good_crossed * (6.5 * d_next)

            # Bad crossing is not just high cost; mark it unsafe for risk selection.
            min_margin = np.where(bad_crossed, np.minimum(min_margin, -0.10), min_margin)

        # Numerically safe output.
        cost = np.nan_to_num(cost, nan=1e12, posinf=1e12, neginf=1e12)
        min_margin = np.nan_to_num(min_margin, nan=-1.0, posinf=1e6, neginf=-1.0)
        return cost, min_margin, best_stage_positions, good_crossed, bad_crossed

    def _update_distribution(
        self,
        samples: np.ndarray,
        cost: np.ndarray,
        min_margin: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """Exponential weighted refit plus elite covariance adaptation."""
        safe_mask = min_margin > 0.0
        if np.any(safe_mask):
            fit_idx = np.where(safe_mask)[0]
        else:
            fit_idx = np.arange(samples.shape[0])

        c = cost[fit_idx]
        c_min = float(np.min(c))
        weights = np.exp(-(c - c_min) / max(float(self.TEMPERATURE), _EPS))
        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum < _EPS:
            best_idx = int(fit_idx[int(np.argmin(c))])
            weights = np.zeros_like(c)
            weights[int(np.argmin(c))] = 1.0
            w_sum = 1.0
        else:
            best_idx = int(fit_idx[int(np.argmin(c))])

        weights /= w_sum
        new_mean = np.sum(samples[fit_idx] * weights[:, None, None], axis=0)
        new_mean = self._clip_u(new_mean)

        # Adapt covariance using the best few safe samples. Keep a minimum exploration level.
        elite_count = min(int(self.N_ELITES), len(fit_idx))
        elite_local = np.argsort(c)[:elite_count]
        elite_idx = fit_idx[elite_local]
        elite_delta = samples[elite_idx] - new_mean[None, :, :]
        elite_std = np.sqrt(np.mean(elite_delta ** 2, axis=0) + 1e-8)
        target_sigma = np.clip(1.35 * elite_std, self.SIGMA_MIN, self.SIGMA_MAX)
        self._sigma = np.clip(0.82 * self._sigma + 0.18 * target_sigma,
                              self.SIGMA_MIN, self.SIGMA_MAX)
        self._mean_u = new_mean

        return new_mean, best_idx, bool(np.any(safe_mask))

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------
    def _reactive_push(self, pos: np.ndarray) -> np.ndarray:
        """Small final safety push; the optimizer is still the main planner."""
        push = np.zeros(3, dtype=np.float64)
        for op in self._obstacle_positions:
            diff = pos[:2] - op[:2]
            d = float(np.linalg.norm(diff))
            if _EPS < d < self.APF_INFLUENCE:
                mag = self.APF_GAIN * (1.0 / d - 1.0 / self.APF_INFLUENCE) / (d * d)
                mag = min(float(mag), self.APF_MAX)
                push[:2] += mag * diff / d

        # When close to the current gate plane, reactive obstacle pushes must
        # not push the vehicle farther toward the frame. Work in gate-local
        # coordinates: any reactive y/z component that increases |local_y| or
        # |local_z| is strongly damped, then an inward centering correction is
        # added. This prevents pole APF from pushing the drone into the gate bar.
        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            R = self._gate_rotmats[self._target_gate]
            local = R.T @ (pos - gp)
            x_abs = abs(float(local[0]))
            if x_abs < self.FUNNEL_LENGTH:
                alpha = 1.0 - x_abs / self.FUNNEL_LENGTH
                local_push = R.T @ push
                if float(local[1] * local_push[1]) > 0.0:
                    local_push[1] *= 0.12
                if float(local[2] * local_push[2]) > 0.0:
                    local_push[2] *= 0.12
                push = R @ local_push
                y_corr = -float(local[1]) * R[:, 1]
                z_corr = -float(local[2]) * R[:, 2]
                push += 0.30 * alpha * (y_corr + z_corr)

        n = float(np.linalg.norm(push[:2]))
        if n > self.APF_MAX:
            push[:2] *= self.APF_MAX / n
        push[2] = float(np.clip(push[2], -0.5, 0.5))
        return push

    def _yaw_command(self, obs: dict[str, "NDArray[np.floating]"], pos: np.ndarray) -> float:
        current_yaw = float(Rot.from_quat(obs["quat"]).as_euler("xyz")[2])
        if 0 <= self._target_gate < self._n_gates:
            to_gate = self._gate_positions[self._target_gate] - pos
            if float(np.linalg.norm(to_gate[:2])) > 0.08:
                yaw_des = float(np.arctan2(to_gate[1], to_gate[0]))
            else:
                yaw_des = current_yaw
        else:
            yaw_des = current_yaw

        yaw_error = (yaw_des - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        return current_yaw + float(np.clip(0.18 * yaw_error, -0.12, 0.12))

    def _interp_cached_reference(
        self,
        elapsed: float,
        ref_pos: np.ndarray | None = None,
        ref_vel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the selected route reference at elapsed planning time."""
        if ref_pos is None:
            ref_pos = self._cached_ref_pos
        if ref_vel is None:
            ref_vel = self._cached_ref_vel

        if ref_pos is None or ref_vel is None or len(ref_pos) < 2:
            return self._last_pos.copy(), np.zeros(3, dtype=np.float64)

        tau = max(0.0, float(elapsed) + self.TRACK_LOOKAHEAD)
        s = tau / max(self.MPC_DT, _EPS)
        k0 = int(np.clip(math.floor(s), 0, self.MPC_N - 1))
        k1 = min(k0 + 1, self.MPC_N)
        a = float(np.clip(s - k0, 0.0, 1.0))
        p_ref = (1.0 - a) * ref_pos[k0] + a * ref_pos[k1]
        v_ref = (1.0 - a) * ref_vel[k0] + a * ref_vel[k1]
        return p_ref.astype(np.float64), v_ref.astype(np.float64)

    def _project_polyline_tracking_target(
        self,
        polyline: np.ndarray,
        pos: np.ndarray,
        lookahead_dist: float,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Project the drone onto a polyline and return a lookahead point and tangent.

        Time-indexed tracking is weak in sharp turns: if the real drone is slightly
        late or wide, the time reference can be behind/sideways and does not pull
        hard enough back to the intended centerline. Projection makes the tracker
        spatial: first find the closest point on the selected path, then track a
        point ahead along that same path.
        """
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return pos.copy(), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0, 0.0

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        valid = seg_len > 1e-6
        if not np.any(valid):
            return pts[0].copy(), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0, 0.0

        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        best_d2 = math.inf
        best_i = 0
        best_t = 0.0
        best_s = 0.0

        for i in range(len(seg)):
            if seg_len[i] <= 1e-6:
                continue
            v = seg[i]
            t = float(np.clip(np.dot(pos - pts[i], v) / max(float(np.dot(v, v)), _EPS), 0.0, 1.0))
            q = pts[i] + t * v
            d2 = float(np.dot(pos - q, pos - q))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
                best_s = float(cum[i] + t * seg_len[i])

        target_s = min(best_s + max(float(lookahead_dist), 0.02), float(cum[-1]))
        j = int(np.searchsorted(cum, target_s, side="right") - 1)
        j = int(np.clip(j, 0, len(seg_len) - 1))
        denom = max(float(seg_len[j]), _EPS)
        a = float(np.clip((target_s - cum[j]) / denom, 0.0, 1.0))
        target = (1.0 - a) * pts[j] + a * pts[j + 1]

        tangent = self._unit(seg[j], fallback=seg[best_i])
        return target.astype(np.float64), tangent.astype(np.float64), j, best_s

    def _path_turn_speed_scale(self, polyline: np.ndarray, seg_idx: int) -> float:
        """Return <1 near an upcoming sharp bend, otherwise near 1.

        This is intentionally simple and cheap. It looks a few segments ahead and
        reduces desired along-path speed when the selected path bends sharply.
        That gives the route tracker authority to brake before the drone drifts
        outward through a corner.
        """
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 4:
            return 1.0
        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        dirs = []
        for i in range(len(seg)):
            if seg_len[i] > 1e-6:
                dirs.append(seg[i] / seg_len[i])
            else:
                dirs.append(None)

        max_angle = 0.0
        start = int(np.clip(seg_idx, 0, len(seg) - 1))
        stop = min(start + 4, len(seg) - 1)
        for i in range(start, stop):
            a = dirs[i]
            b = dirs[i + 1]
            if a is None or b is None:
                continue
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            max_angle = max(max_angle, math.acos(dot))

        if max_angle <= self.TURN_ANGLE_FOR_SLOWDOWN:
            return 1.0
        turn_level = np.clip(
            (max_angle - self.TURN_ANGLE_FOR_SLOWDOWN) / max(math.pi * 0.75 - self.TURN_ANGLE_FOR_SLOWDOWN, _EPS),
            0.0,
            1.0,
        )
        return float(1.0 - 0.42 * turn_level)

    def _route_tracking_feedback(
        self,
        u_base: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        elapsed: float,
        ref_pos: np.ndarray | None = None,
        ref_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Closed-loop correction toward the selected route reference.

        The scenario sequence is the feedforward plan. This feedback is the
        reactive part that makes the real drone follow the displayed route in
        sharp turns. It uses projection onto the selected path, strong lateral
        damping, and automatic braking when the drone is wide or a turn is ahead.
        """
        if ref_pos is None:
            ref_pos = self._cached_ref_pos
        if ref_vel is None:
            ref_vel = self._cached_ref_vel

        # Prefer tracking the deterministic anchor rollout when available because
        # it is usually the blue/cyan line you visually want to follow. Fall back
        # to the selected reference path, then to time-indexed reference tracking.
        path = None
        if self.USE_PROJECTED_ROUTE_TRACKING:
            if self._last_route_anchor_traj is not None and len(self._last_route_anchor_traj) >= 3:
                path = np.asarray(self._last_route_anchor_traj, dtype=np.float64)
            elif ref_pos is not None and len(ref_pos) >= 3:
                path = np.asarray(ref_pos, dtype=np.float64)
            elif self._active_route_points is not None and len(self._active_route_points) >= 2:
                path = np.asarray(self._active_route_points, dtype=np.float64)

        if path is None or len(path) < 2:
            p_ref, v_ref = self._interp_cached_reference(elapsed, ref_pos, ref_vel)
            e_p = p_ref - pos
            e_v = v_ref - vel
            corr = self.TRACK_KP * e_p + self.TRACK_KD * e_v
        else:
            speed = float(np.linalg.norm(vel))
            lookahead = self.PATH_LOOKAHEAD_DIST + self.PATH_LOOKAHEAD_SPEED_GAIN * speed
            lookahead = float(np.clip(lookahead, self.PATH_LOOKAHEAD_DIST, self.PATH_LOOKAHEAD_MAX))
            p_ref, tangent, seg_idx, _ = self._project_polyline_tracking_target(path, pos, lookahead)

            e = p_ref - pos
            e_parallel = tangent * float(np.dot(e, tangent))
            e_perp = e - e_parallel

            v_parallel_scalar = float(np.dot(vel, tangent))
            v_parallel = tangent * v_parallel_scalar
            v_perp = vel - v_parallel

            turn_scale = self._path_turn_speed_scale(path, seg_idx)
            cross_xy = float(np.linalg.norm(e_perp[:2]))
            if cross_xy <= self.CROSS_TRACK_SLOWDOWN_START:
                cross_scale = 1.0
            else:
                cross_scale = 1.0 - 0.45 * float(np.clip(
                    (cross_xy - self.CROSS_TRACK_SLOWDOWN_START)
                    / max(self.CROSS_TRACK_SLOWDOWN_FULL - self.CROSS_TRACK_SLOWDOWN_START, _EPS),
                    0.0,
                    1.0,
                ))

            desired_speed = self.V_CRUISE * turn_scale * cross_scale
            desired_speed = float(np.clip(desired_speed, self.TURN_SPEED_MIN, self.TURN_SPEED_MAX))

            # Near the active gate, do not blast through a lateral error. The
            # frame is unforgiving, so use the gate passage speed as an upper cap.
            if 0 <= self._target_gate < self._n_gates:
                d_gate = float(np.linalg.norm(pos - self._gate_positions[self._target_gate]))
                if d_gate < self.ALIGN_START_DIST:
                    gate_blend = 1.0 - d_gate / max(self.ALIGN_START_DIST, _EPS)
                    desired_speed = min(desired_speed, self.V_CRUISE - gate_blend * (self.V_CRUISE - self.V_GATE))

            desired_v_parallel = tangent * desired_speed

            corr = (
                self.CROSS_TRACK_KP * e_perp
                - self.CROSS_TRACK_KD * v_perp
                + self.ALONG_TRACK_KP * e_parallel
                + self.ALONG_TRACK_KD * (desired_v_parallel - v_parallel)
            )

            # Extra braking if the drone is too fast along the path while already
            # wide or entering a turn. This is what reduces outward corner drift.
            turn_or_wide = max(1.0 - turn_scale, 0.0) + min(cross_xy / 0.30, 1.0)
            overspeed = max(v_parallel_scalar - desired_speed, 0.0)
            if overspeed > 0.0 and turn_or_wide > 0.05:
                corr -= self.TURN_BRAKE_GAIN * turn_or_wide * overspeed * tangent

            # Keep vertical correction less aggressive than horizontal route correction.
            corr[2] = self.TRACK_KP[2] * e[2] - self.TRACK_KD[2] * float(vel[2])

        xy_norm = float(np.linalg.norm(corr[:2]))
        if xy_norm > self.TRACK_MAX_XY:
            corr[:2] *= self.TRACK_MAX_XY / max(xy_norm, _EPS)
        corr[2] = float(np.clip(corr[2], -self.TRACK_MAX_Z, self.TRACK_MAX_Z))
        return self._clip_u(np.asarray(u_base, dtype=np.float64) + corr)

    def _candidate_progress_cost(
        self,
        indices: np.ndarray,
        cost: np.ndarray,
        trajectories: np.ndarray,
        pos: np.ndarray,
        min_margin: np.ndarray,
        cross_yz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cost used only for selecting among fresh non-crashing candidates."""
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)

        select_cost = cost[indices].astype(np.float64).copy()

        # Prefer real extra clearance. This dominates tiny differences in cost
        # and prevents frame-skimming rollouts from winning just because they are
        # slightly shorter/faster.
        select_cost += 9000.0 * np.maximum(self.SELECTION_MARGIN - min_margin[indices], 0.0) ** 2
        select_cost -= 140.0 * np.maximum(min_margin[indices] - self.SELECTION_MARGIN, 0.0)

        # If the candidate crosses the gate, prefer crossing near the center.
        if cross_yz is not None:
            yz = cross_yz[indices]
            finite = np.isfinite(yz)
            select_cost[finite] += 700.0 * (yz[finite] / max(self.ROBUST_CROSS_RADIUS, _EPS)) ** 2

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            n = self._get_gate_normal(self._target_gate, pos)
            signed0 = float((pos - gp) @ n)
            final = trajectories[indices, -1, :].astype(np.float64)
            progress = (final - gp) @ n - signed0
            d_final = np.linalg.norm(final - gp, axis=1)
            select_cost -= 620.0 * progress
            select_cost += 115.0 * d_final

        return select_cost

    def _choose_fresh_robust_candidates(
        self,
        cost: np.ndarray,
        min_margin: np.ndarray,
        trajectories: np.ndarray,
        good_crossed: np.ndarray,
        bad_crossed: np.ndarray,
        pos: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Choose the best actual fresh rollout; do not average unsafe modes.

        This is the critical gate-0 crash fix. Candidate rollouts are filtered in
        buckets. The normal command source must be non-crashing and robust. If a
        gate-crossing rollout exists, it must also cross near the opening center,
        not merely barely inside the frame. The returned set is tiny; with
        ELITE_BLEND_COUNT=1, the applied sequence is an actual simulated rollout,
        not the mean of several rollouts.
        """
        cross_yz, geometric_crossed = self._crossing_yz_from_trajectories(trajectories, pos)

        noncrash = (~bad_crossed) & (min_margin > self.SAFE_SELECTION_MARGIN)
        robust = (~bad_crossed) & (min_margin > self.SELECTION_MARGIN)

        robust_center_cross = robust & good_crossed & geometric_crossed & (cross_yz <= self.ROBUST_CROSS_RADIUS)
        robust_any_cross = robust & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS)
        loose_center_cross = noncrash & good_crossed & geometric_crossed & (cross_yz <= self.ROBUST_CROSS_RADIUS)
        loose_any_cross = noncrash & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS)

        if np.any(robust_center_cross):
            pool = np.where(robust_center_cross)[0]
            label = "robust_center_cross"
        elif np.any(robust_any_cross):
            pool = np.where(robust_any_cross)[0]
            label = "robust_cross"
        elif np.any(loose_center_cross):
            pool = np.where(loose_center_cross)[0]
            label = "loose_center_cross"
        elif np.any(loose_any_cross):
            pool = np.where(loose_any_cross)[0]
            label = "loose_cross"
        elif np.any(robust):
            pool = np.where(robust)[0]
            label = "robust_progress"
        elif np.any(noncrash):
            pool = np.where(noncrash)[0]
            label = "loose_progress"
        else:
            # Last-resort fallback: every scenario is unsafe. Choose the largest
            # margin, then best progress/cost. This should be rare and visible in
            # the debug mode label.
            pool = np.arange(cost.shape[0])
            label = "emergency"

        sel_cost = self._candidate_progress_cost(pool, cost, trajectories, pos, min_margin, cross_yz)
        if label == "emergency":
            order = pool[np.lexsort((sel_cost, -min_margin[pool]))]
        else:
            order = pool[np.argsort(sel_cost)]

        keep_n = max(1, min(int(self.ELITE_BLEND_COUNT), len(order)))
        return order[:keep_n].astype(int), label

    def _elite_weighted_sequence(
        self,
        samples: np.ndarray,
        cost: np.ndarray,
        candidate_idx: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Return an exponential-cost mean of the best robust candidate scenarios."""
        if candidate_idx.size == 0:
            best_idx = int(np.argmin(cost))
            return self._clip_u(samples[best_idx].copy()), best_idx, 1

        order = candidate_idx[np.argsort(cost[candidate_idx])]
        elite = order[:max(1, min(int(self.ELITE_BLEND_COUNT), len(order)))]
        c = cost[elite]
        c0 = float(np.min(c))
        w = np.exp(-(c - c0) / max(float(self.ELITE_TEMPERATURE), _EPS))
        ws = float(np.sum(w))
        if not np.isfinite(ws) or ws < _EPS:
            w = np.zeros_like(c)
            w[0] = 1.0
            ws = 1.0
        w /= ws
        seq = np.sum(samples[elite] * w[:, None, None], axis=0)
        return self._clip_u(seq), int(elite[0]), int(len(elite))


    def _crossing_yz_from_trajectories(
        self,
        trajectories: np.ndarray,
        pos0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate gate-crossing center offset for each rollout trajectory.

        Returns:
            yz_radius: sqrt(local_y^2 + local_z^2) at first current-gate crossing.
                       inf if the trajectory never crosses.
            crossed: boolean crossing flag from the sampled trajectory geometry.

        This is used for selection only. It prevents choosing a trajectory that is
        technically non-crashing but crosses close to the frame, where model
        mismatch can still cause a hit.
        """
        K = trajectories.shape[0]
        yz_radius = np.full(K, np.inf, dtype=np.float64)
        crossed = np.zeros(K, dtype=bool)

        if not (0 <= self._target_gate < self._n_gates):
            return yz_radius, crossed

        gp = self._gate_positions[self._target_gate]
        R = self._gate_rotmats[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos0)
        pts = trajectories.astype(np.float64, copy=False)
        signed = (pts - gp) @ normal

        for k in range(pts.shape[1] - 1):
            s0 = signed[:, k]
            s1 = signed[:, k + 1]
            mask = (~crossed) & (s0 <= 0.0) & (s1 >= 0.0)
            if not np.any(mask):
                continue
            denom = np.maximum(s1[mask] - s0[mask], _EPS)
            alpha = np.clip(-s0[mask] / denom, 0.0, 1.0)
            p_cross = pts[mask, k, :] + alpha[:, None] * (pts[mask, k + 1, :] - pts[mask, k, :])
            local = self._world_to_gate_local(p_cross, gp, R)
            yz_radius[mask] = np.sqrt(local[:, 1] * local[:, 1] + local[:, 2] * local[:, 2])
            crossed[mask] = True

        return yz_radius, crossed

    def _finish_action_from_u(
        self,
        obs: dict[str, "NDArray[np.floating]"],
        pos: np.ndarray,
        u0: np.ndarray,
    ) -> "NDArray[np.floating]":
        """Convert the selected specific-thrust command to the simulator action."""
        yaw_cmd = self._yaw_command(obs, pos)
        output = accel_to_attitude(
            a_cmd=u0,
            yaw_des=yaw_cmd,
            mass=self._mass_estimate,
            max_tilt=self.MAX_TILT_CMD,
            thrust_min=self._thrust_min,
            thrust_max=self._thrust_max,
        )
        roll_cmd, pitch_cmd, yaw_out, thrust_cmd = output

        roll_cmd = float(np.clip(roll_cmd, -self.MAX_TILT_CMD, self.MAX_TILT_CMD))
        pitch_cmd = float(np.clip(pitch_cmd, -self.MAX_TILT_CMD, self.MAX_TILT_CMD))

        # Recompute thrust once more after final roll/pitch clipping so the
        # vertical component remains the commanded one.
        thrust_cmd = _vertical_preserving_thrust(
            az_world=float(u0[2]),
            roll=roll_cmd,
            pitch=pitch_cmd,
            mass=self._mass_estimate,
            thrust_min=self._thrust_min,
            thrust_max=self._thrust_max,
        )

        # Last physical safety clamps. These are deliberately applied after the
        # vertical-preserving thrust calculation.
        if pos[2] < self.GROUND_CLEARANCE + 0.08:
            thrust_cmd += 0.65 * (self.GROUND_CLEARANCE + 0.08 - pos[2]) * self._mass_estimate * self._g
        if pos[2] > self.CEILING - 0.18:
            overshoot = pos[2] - (self.CEILING - 0.18)
            thrust_cmd -= (0.55 + 2.2 * overshoot) * overshoot * self._mass_estimate * self._g

        thrust_cmd = float(np.clip(thrust_cmd, self._thrust_min, self._thrust_max))

        self._prev_output = np.array([roll_cmd, pitch_cmd, yaw_out, thrust_cmd], dtype=np.float64)
        return self._prev_output.astype(np.float32)

    # -------------------------------------------------------------------------
    # Main controller API
    # -------------------------------------------------------------------------
    def compute_control(
        self,
        obs: dict[str, "NDArray[np.floating]"],
        info: dict | None = None,
    ) -> "NDArray[np.floating]":
        self._tick += 1
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        self._last_pos = pos.copy()

        if self._launch_z is None:
            self._launch_z = float(pos[2])

        if self._finished:
            return np.array([0.0, 0.0, 0.0, self._mass_estimate * self._g], dtype=np.float32)

        # Some environments update target_gate directly in obs before step_callback.
        if "target_gate" in obs:
            obs_target_gate = int(obs["target_gate"])
            if obs_target_gate != self._target_gate:
                self._sigma = np.maximum(self._sigma, self.SIGMA_INIT[None, :])
                self._last_planner_tick = -10**9
                self._cached_plan_tick = -10**9
                self._cached_u_index = 0
                self._active_route_points = None
                self._last_best_traj = None
                self._last_route_anchor_traj = None
            self._target_gate = obs_target_gate
            if self._target_gate < 0:
                self._finished = True
                return np.array([0.0, 0.0, 0.0, self._mass_estimate * self._g], dtype=np.float32)

        do_replan = (self._tick - self._last_planner_tick) >= self._planner_interval_steps

        # Fast path: reuse the selected sequence from the last expensive replan.
        # This is used only when the simulator/control callback is faster than
        # the requested 50 Hz planner rate. Index by elapsed physical time, not
        # by callback count; otherwise a 500 Hz simulator would burn through a
        # 0.09 s MPC step every 0.002 s and the cached plan would advance 45x too fast.
        if not do_replan:
            elapsed_since_plan = max(0.0, (self._tick - self._cached_plan_tick) * self._dt)
            idx = min(int(elapsed_since_plan / max(self.MPC_DT, _EPS)), self.MPC_N - 1)
            u0 = self._cached_u_sequence[idx].copy()
            self._cached_u_index = idx
            u0 = self._route_tracking_feedback(u0, pos, vel, elapsed_since_plan)

            # Still apply cheap safety/altitude corrections every physics step.
            u0 = self.CMD_FILTER_ALPHA * u0 + (1.0 - self.CMD_FILTER_ALPHA) * self._prev_accel
            u0 += self._reactive_push(pos)
            u0 = self._limit_commanded_accel(u0)
            u0 = self._launch_altitude_guard(u0, pos, vel)
            u0 = self._limit_commanded_accel(u0)

            self._prev_accel = u0.copy()
            self._cached_u0 = u0.copy()

            if self._tick % 250 == 0:
                speed = float(np.linalg.norm(vel))
                print(
                    f"[SCENARIO-MPC] step={self._tick:04d} cached "
                    f"gate={self._target_gate} v={speed:.2f} "
                    f"idx={idx}/{self.MPC_N - 1} u=[{u0[0]:+.2f},{u0[1]:+.2f},{u0[2]:+.2f}]"
                )

            return self._finish_action_from_u(obs, pos, u0)

        plan_t0 = time.perf_counter()
        previous_plan_tick = self._cached_plan_tick
        self._last_planner_tick = self._tick

        # In fresh-scenario mode the next plan is rebuilt from the current state
        # and current route anchors. Do not shift old MPPI means into the new
        # candidate distribution; old near-frame or outward-turn modes caused the
        # gate-0 crash. If fresh mode is disabled, keep the classic warm-start.
        if not self.FORGET_OLD_SCENARIOS:
            if previous_plan_tick > -10**8:
                elapsed_since_previous_plan = max(0.0, (self._tick - previous_plan_tick) * self._dt)
                shift_steps = min(int(elapsed_since_previous_plan / max(self.MPC_DT, _EPS)), self.MPC_N - 1)
            else:
                shift_steps = 0
            for _ in range(shift_steps):
                self._shift_distribution()

        # First build many deterministic route candidates, then rank them by a
        # complete forward rollout of their PD anchor. This avoids the failure mode
        # where a safe route is drawn/found but not selected because the cheap
        # route score or the MPPI mean prefers a sideways/hovering mode.
        raw_routes, raw_route_costs = self._build_route_candidates(pos, vel)
        routes, route_costs, route_seqs, ref_pos, ref_vel = self._rank_routes_by_rollout(
            pos, vel, raw_routes, raw_route_costs,
        )

        self._route_candidates = routes
        self._route_costs = route_costs
        self._active_route_points = routes[0].copy() if routes else None

        # Fresh planning: sample around the current route anchors. The old mean
        # is not allowed to bias scenario generation when FORGET_OLD_SCENARIOS is
        # enabled. This prevents stale scenarios from dragging the command into
        # the gate frame.
        if route_seqs:
            if self.FORGET_OLD_SCENARIOS:
                self._mean_u = self._clip_u(route_seqs[0].copy())
            else:
                self._mean_u = self._clip_u(
                    (1.0 - self.ROUTE_MEAN_BLEND) * self._mean_u
                    + self.ROUTE_MEAN_BLEND * route_seqs[0]
                )

        samples = self._sample_sequences(route_seqs)

        cost, min_margin, trajectories, good_crossed, bad_crossed = self._score_rollouts(
            samples, pos, vel, ref_pos, ref_vel,
        )
        new_mean, best_idx, any_safe = self._update_distribution(samples, cost, min_margin)

        # Choose only fresh, non-crashing candidates from THIS planning tick.
        # Do not let old MPPI means or frame-grazing samples enter the applied
        # sequence. With ELITE_BLEND_COUNT=1 this is the single best robust
        # scenario; if you later raise ELITE_BLEND_COUNT, it averages only the
        # top tiny set returned here.
        candidate_idx, selection_label = self._choose_fresh_robust_candidates(
            cost, min_margin, trajectories, good_crossed, bad_crossed, pos,
        )
        selected_sequence, selected_idx, elite_count = self._elite_weighted_sequence(samples, cost, candidate_idx)

        # Mean-rollout evaluation is optional. The applied command is now scenario-committed,
        # so evaluating the MPPI mean every 50 Hz plan is mostly a debug feature. Skipping it
        # removes one extra horizon rollout and reduces solve-time spikes.
        if self.EVALUATE_MEAN_ROLLOUT:
            mean_cost, mean_margin, mean_traj, _, _ = self._score_rollouts(
                new_mean[None, :, :], pos, vel, ref_pos, ref_vel,
            )
            mean_margin0 = float(mean_margin[0])
            self._last_mean_traj = mean_traj[0].astype(np.float64)
        else:
            mean_margin0 = self.RISK_BLEND_MARGIN + 1.0
            self._last_mean_traj = None

        chosen_u0 = selected_sequence[0].copy()
        route_u0 = route_seqs[0][0].copy() if route_seqs else chosen_u0.copy()
        best_margin = float(min_margin[selected_idx])
        robust_selected = best_margin > self.SELECTION_MARGIN and not bool(bad_crossed[selected_idx])

        # Apply the selected fresh robust scenario. Do not mix in the MPPI mean
        # and do not mix in an unsafe global best. The previous elite/mean blend
        # could average two non-crashing routes into a crashing middle route at
        # gate 0. The route anchor only contributes a small stabilizing bias.
        if robust_selected:
            u0 = 0.90 * chosen_u0 + 0.10 * route_u0
        elif min_margin[selected_idx] > 0.0 and not bool(bad_crossed[selected_idx]):
            u0 = 0.82 * chosen_u0 + 0.18 * route_u0
        else:
            # Emergency: if every scenario is unsafe, prefer the deterministic
            # route tracker and let the reactive/altitude safety layers act.
            u0 = route_u0.copy()

        if route_seqs:
            u0 = (1.0 - self.ROUTE_COMMAND_BLEND) * u0 + self.ROUTE_COMMAND_BLEND * route_u0

        # Add closed-loop route tracking before output smoothing. This is the main
        # anti-wide-turn fix: the plan remains a feedforward scenario, but the
        # actual command corrects toward the selected reference at every callback.
        u0 = self._route_tracking_feedback(u0, pos, vel, 0.0, ref_pos, ref_vel)

        # Commit the warm-start to the selected fresh sequence only. In fresh
        # mode this is used mainly for debug/fallback; the next planning tick
        # still samples from current route anchors instead of old scenario means.
        self._mean_u = self._clip_u(selected_sequence.copy() if self.FORGET_OLD_SCENARIOS
                                    else 0.70 * selected_sequence + 0.30 * new_mean)

        # Cache the full selected sequence and execute it between expensive
        # planner calls if compute_control is called faster than 50 Hz. Indexing
        # is time-based in the fast path, so the first command is held until
        # elapsed time reaches MPC_DT.
        self._cached_u_sequence = selected_sequence.copy()
        self._cached_ref_pos = ref_pos.copy()
        self._cached_ref_vel = ref_vel.copy()
        self._cached_plan_tick = self._tick
        self._cached_u_index = 0
        self._cached_u0 = selected_sequence[0].copy()
        self._last_elite_count = int(elite_count)

        # Smooth against the actually applied previous command.
        u0 = self.CMD_FILTER_ALPHA * u0 + (1.0 - self.CMD_FILTER_ALPHA) * self._prev_accel
        u0 += self._reactive_push(pos)
        u0 = self._limit_commanded_accel(u0)
        u0 = self._launch_altitude_guard(u0, pos, vel)
        u0 = self._limit_commanded_accel(u0)

        self._prev_accel = u0.copy()
        self._last_selected_idx = int(selected_idx)
        self._last_best_cost = float(cost[selected_idx])
        self._last_best_margin = best_margin
        self._last_best_traj = trajectories[selected_idx].astype(np.float64)

        self._last_plan_ms = 1000.0 * (time.perf_counter() - plan_t0)
        self._plan_counter += 1
        if self._plan_counter == 1:
            self._plan_ms_ema = self._last_plan_ms
        else:
            self._plan_ms_ema = 0.90 * self._plan_ms_ema + 0.10 * self._last_plan_ms

        # Debug output.
        if self._plan_counter % max(1, int(self.DEBUG_EVERY_PLANS)) == 0:
            speed = float(np.linalg.norm(vel))
            vz = float(vel[2])
            if 0 <= self._target_gate < self._n_gates:
                gp = self._gate_positions[self._target_gate]
                gate_dist = float(np.linalg.norm(pos - gp))
                dz = float(gp[2] - pos[2])
                R = self._gate_rotmats[self._target_gate]
                local = R.T @ (pos - gp)
                gate_yz = math.sqrt(float(local[1] * local[1] + local[2] * local[2]))
            else:
                gate_dist = -1.0
                dz = 0.0
                gate_yz = 0.0
            print(
                f"[SCENARIO-MPC] step={self._tick:04d} gate={self._target_gate} "
                f"pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] "
                f"v={speed:.2f} vz={vz:+.2f} dz={dz:+.2f} "
                f"d_gate={gate_dist:.2f} yz={gate_yz:.3f} "
                f"route_cost={self._last_route_cost:.2f} "
                f"anchor_cost={self._last_route_anchor_cost:.1f} "
                f"anchor_cross={self._last_route_anchor_crossed} "
                f"best_cost={self._last_best_cost:.1f} margin={self._last_best_margin:.3f} "
                f"safe={any_safe} crossed={bool(good_crossed[selected_idx])} "
                f"sel={self._last_selected_idx} elite={self._last_elite_count} mode={selection_label} "
                f"plan_ms={self._last_plan_ms:.1f}/{self._plan_ms_ema:.1f} "
                f"u=[{u0[0]:+.2f},{u0[1]:+.2f},{u0[2]:+.2f}]"
            )

        return self._finish_action_from_u(obs, pos, u0)

    def step_callback(
        self,
        action: "NDArray[np.floating]",
        obs: dict[str, "NDArray[np.floating]"],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        new_target = int(obs["target_gate"])
        gate_changed = new_target != self._target_gate
        self._target_gate = new_target

        if gate_changed:
            self._sigma = np.maximum(self._sigma, self.SIGMA_INIT[None, :])
            self._last_planner_tick = -10**9
            self._cached_plan_tick = -10**9
            self._cached_u_index = 0
            self._active_route_points = None
            self._last_best_traj = None
            self._last_route_anchor_traj = None

        if new_target < 0 or terminated or truncated:
            self._finished = True
            return True

        # Update sensed gate poses once they become available/visited.
        for i in range(self._n_gates):
            if obs["gates_visited"][i] and not self._gates_visited[i]:
                self._gates_visited[i] = True
                self._gate_positions[i] = np.array(obs["gates_pos"][i], dtype=np.float64)
                self._gate_quats[i] = np.array(obs["gates_quat"][i], dtype=np.float64)
                self._gate_rotmats[i] = Rot.from_quat(self._gate_quats[i]).as_matrix()

        for i in range(len(self._obstacle_positions)):
            if obs["obstacles_visited"][i] and not self._obstacles_visited[i]:
                self._obstacles_visited[i] = True
                self._obstacle_positions[i] = np.array(obs["obstacles_pos"][i], dtype=np.float64)

        # Safer online mass estimate: only update near hover. Updating during
        # aggressive tilted flight tends to overestimate mass and increases climb.
        if len(action) >= 4 and self._tick > 10:
            roll = float(action[0])
            pitch = float(action[1])
            thrust = float(action[3])
            vel = np.asarray(obs["vel"], dtype=np.float64)

            near_hover_attitude = abs(roll) < 0.15 and abs(pitch) < 0.15
            near_hover_vertical = abs(float(vel[2])) < 0.25

            if near_hover_attitude and near_hover_vertical and thrust > 0.01:
                vertical_factor = max(math.cos(roll) * math.cos(pitch), 0.25)
                mass_obs = thrust * vertical_factor / self._g
                if 0.5 * self._mass < mass_obs < 1.8 * self._mass:
                    self._mass_estimate = 0.995 * self._mass_estimate + 0.005 * mass_obs

        return self._finished

    def render_callback(self, sim: "Sim"):
        from crazyflow.sim.visualize import draw_line, draw_points

        # Draw route/scenario lines every render frame. The line primitives are not
        # persistent in the viewer, so returning here makes them blink/disappear.
        # Heavy auxiliary geometry is gated below by DRAW_FULL_DEBUG_GEOMETRY.
        drone_pos = getattr(self, "_last_pos", np.zeros(3, dtype=np.float64))

        # Draw selected deterministic route and best sampled trajectory from the previous MPC step.
        if self._active_route_points is not None and len(self._active_route_points) > 1:
            draw_line(sim, self._active_route_points, rgba=(1.0, 0.85, 0.1, 0.90),
                      start_size=2.2, end_size=2.2)

        if self._last_best_traj is not None and len(self._last_best_traj) > 1:
            draw_line(sim, self._last_best_traj, rgba=(0.0, 1.0, 0.2, 0.85),
                      start_size=2.0, end_size=2.0)

        # Cyan: deterministic route-anchor rollout selected before stochastic sampling.
        if self._last_route_anchor_traj is not None and len(self._last_route_anchor_traj) > 1:
            draw_line(sim, self._last_route_anchor_traj, rgba=(0.0, 0.9, 1.0, 0.50),
                      start_size=1.5, end_size=1.5)

        # Purple: refitted MPPI mean rollout. If this differs strongly from green,
        # averaging was the problem; the applied command now follows green/anchor.
        if self._last_mean_traj is not None and len(self._last_mean_traj) > 1:
            draw_line(sim, self._last_mean_traj, rgba=(1.0, 0.0, 1.0, 0.38),
                      start_size=1.3, end_size=1.3)

        # Current target.
        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            draw_points(sim, gp.reshape(1, -1), rgba=(1.0, 1.0, 0.0, 1.0), size=0.04)
            draw_line(sim, np.vstack([drone_pos, gp]), rgba=(0.0, 1.0, 0.0, 0.55),
                      start_size=2.0, end_size=2.0)

        if not self.DRAW_FULL_DEBUG_GEOMETRY:
            return

        # Pole safety circles.
        vis_range = 1.7
        for op in self._obstacle_positions:
            if float(np.linalg.norm(drone_pos[:2] - op[:2])) > vis_range:
                continue
            n_ring = 32
            angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
            ring = np.zeros((n_ring + 1, 3), dtype=np.float64)
            ring[:-1, 0] = op[0] + self.OBSTACLE_RADIUS * np.cos(angles)
            ring[:-1, 1] = op[1] + self.OBSTACLE_RADIUS * np.sin(angles)
            ring[:-1, 2] = 0.5
            ring[-1] = ring[0]
            draw_line(sim, ring, rgba=(1.0, 0.0, 0.0, 0.75), start_size=2.0, end_size=2.0)

        # Gate opening, frame bars, and funnel.
        for gi in range(self._n_gates):
            gp = self._gate_positions[gi]
            R = self._gate_rotmats[gi]
            is_active = gi >= self._target_gate
            alpha = 0.75 if gi == self._target_gate else (0.32 if is_active else 0.16)

            h = self.GATE_HALF_OPENING
            opening_local = np.array([
                [0.0, -h, -h], [0.0, h, -h], [0.0, h, h],
                [0.0, -h, h], [0.0, -h, -h],
            ])
            draw_line(sim, opening_local @ R.T + gp, rgba=(0.2, 0.5, 1.0, alpha),
                      start_size=2.0, end_size=2.0)

            oh = self.GATE_OUTER_HALF
            frame_local = np.array([
                [0.0, -oh, -oh], [0.0, oh, -oh], [0.0, oh, oh],
                [0.0, -oh, oh], [0.0, -oh, -oh],
            ])
            draw_line(sim, frame_local @ R.T + gp, rgba=(1.0, 0.2, 0.2, 0.35 * alpha),
                      start_size=2.0, end_size=2.0)

            if gi == self._target_gate:
                fl = self.FUNNEL_LENGTH
                fo = self.FUNNEL_OUTER_HALF
                for x_off in [-fl, -0.5 * fl, 0.0, 0.5 * fl, fl]:
                    frac = abs(x_off) / max(fl, _EPS)
                    hw = h + frac * (fo - h)
                    rect = np.array([
                        [x_off, -hw, -hw], [x_off, hw, -hw], [x_off, hw, hw],
                        [x_off, -hw, hw], [x_off, -hw, -hw],
                    ])
                    draw_line(sim, rect @ R.T + gp, rgba=(0.2, 0.7, 1.0, 0.20),
                              start_size=1.5, end_size=1.5)