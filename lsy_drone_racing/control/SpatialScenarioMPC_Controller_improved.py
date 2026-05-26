"""Spatial Scenario MPC Controller — MPPI planner with full RPY+thrust dynamics
and spatial curvilinear (Bishop-frame) coordinate system.

Combines the stochastic scenario / MPPI planning framework with:
  - Full rotational dynamics model (roll, pitch, yaw + angular rates + thrust)
  - Spatial curvilinear coordinate system via Parallel Transport (Bishop) frame
  - Dynamic flight corri dor constraints projected onto the transverse plane
  - Curvature-based velocity profiling for aggressive yet safe cornering

Control space:
    u = [phi_c, theta_c, psi_c, T_c]  (roll command, pitch command, yaw command, thrust)
    The rotational dynamics are modelled as a fitted second-order response:
        ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy
    Translational acceleration uses the full rotation matrix:
        acc_world = g + R_IB @ [0, 0, T/m]

The simulator action output is directly [roll, pitch, yaw, thrust] without
any lossy accel_to_attitude conversion.

References:
    Chan et al., "Near time-optimal trajectory optimisation for drones in
    last-mile delivery using spatial reformulation approach", TRC 171, 2025.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from drone_models.core import load_params
from lsy_drone_racing.control.controller import Controller

# Spatial geometry engine (Bishop frame, corridors)
try:
    from lsy_drone_racing.control.GeometryEngines.geometryEngine import GeometryEngine
    from lsy_drone_racing.utils.utils import draw_line

    _HAS_SPATIAL = True
except ImportError as _imp_err:
    _HAS_SPATIAL = False
    print(f"[SPATIAL-IMPORT] Failed: {_imp_err}")

    def draw_line(*a, **kw):
        pass

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_EPS = 1e-9


class SpatialScenarioMPCController(Controller):
    """MPPI / scenario MPC with full RPY+thrust dynamics and spatial curvilinear costs.

    Improvements over PMMRacingController:
      - Full rotational dynamics (RPY + angular rates) in rollout propagation
      - 4-D control space [roll_cmd, pitch_cmd, yaw_cmd, thrust] — direct attitude planning
      - Spatial curvilinear coordinate layer via GeometryEngine (Bishop frame)
      - Corridor violation costs from projected obstacle/gate constraints
      - Curvature-aware velocity profiling
      - No lossy accel-to-attitude post-conversion: the output is what was planned

    Route planning and gate geometry scoring are carried over from the scenario
    MPC framework (deterministic route candidates + MPPI refinement).
    """

    # ========== geometry / task parameters ==========
    # Obstacle poles are 0.03 m in diameter. With the drone half-diagonal
    # (~0.05 m) the geometric clearance needed is ~0.08 m. A radius of 0.25
    # with the 0.08 soft buffer produced a 0.33 m no-go cylinder, which is
    # incompatible with the 0.20 m gate-opening half whenever a pole sits
    # within ~0.4 m of a gate centre (a common env layout) -- the drone
    # cannot fit through both constraints and clips. 0.15 + 0.08 = 0.23 m
    # still leaves a comfortable margin while letting the drone thread
    # gate-adjacent poles.
    # Measured at seed=42, 10 runs: total gates 9 -> 16, finishes 1 -> 2,
    # >=1 gate 4/10 -> 7/10. Visual diagnosis shows fewer obstacle-clip
    # crashes near gates.
    OBSTACLE_RADIUS = 0.15
    OBSTACLE_BUFFER = 0.08

    DRONE_RADIUS = 0.05           # Crazyflie half-diagonal (~0.047) + small margin

    GATE_HALF_OPENING = 0.20      # actual inner edge of frame bars (0.28 - 0.08)
    GATE_OUTER_HALF = 0.28
    GATE_POST_RADIUS = 0.0        # box distance: 0 = touching surface
    GATE_FRAME_BUFFER = 0.12      # soft buffer zone around box surfaces
    GATE_CLEARANCE = 0.045
    GATE_PLANE_SLAB = 0.16

    # Funnel pulls the drone onto the gate centreline over the last
    # FUNNEL_LENGTH metres of approach. 0.70 was short enough that the
    # drone reached the opening height still laterally off the normal;
    # 1.00 was long enough that the planner over-centred and stuck on the
    # final gate (timeouts on random tracks). 0.85 is the middle ground.
    FUNNEL_LENGTH = 0.85
    # The funnel half-width = the actual opening half (0.20). Previously this
    # was 0.28 -- the same as the frame outer half -- which meant the funnel
    # cost gave no penalty for being in the 0.20-0.28 frame-bar band. The
    # drone could "pass" the gate (env in-box is 0.225) yet have its body
    # graze the frame on the way out. Tightening to 0.20 forces the cost to
    # push the drone toward the opening centre.
    FUNNEL_OUTER_HALF = 0.20
    APPROACH_DIST = 0.45
    EXIT_DIST = 0.35
    ALIGN_START_DIST = 1.20

    GROUND_CLEARANCE = 0.10
    CEILING = 1.80

    V_CRUISE = 1.75
    V_GATE = 1.15
    V_MAX = 2.80

    # Attitude / actuator limits
    MAX_ROLL_PITCH_CMD = 0.55       # max roll/pitch command [rad]
    MAX_YAW_CMD = 0.50              # max yaw command magnitude [rad]

    # ========== sampling setup ==========
    # MPC_N 14 * MPC_DT 0.110 = 1.54 s horizon (was 1.10 s). The longer
    # horizon lets the planner see post-gate transitions, which were
    # producing exit-frame clips when the next gate forced a sharp turn.
    MPC_N = 14
    MPC_DT = 0.110
    K_SAMPLES = 150
    N_ELITES = 1
    TEMPERATURE = 50.0
    NOISE_RHO = 0.84
    CMD_FILTER_ALPHA = 0.86
    RISK_BLEND_MARGIN = 0.07
    PLANNER_HZ = 50.0
    RENDER_EVERY = 1
    DRAW_FULL_DEBUG_GEOMETRY = False
    EVALUATE_MEAN_ROLLOUT = True

    SAFE_SELECTION_MARGIN = 0.020
    SELECTION_MARGIN = 0.075
    ROBUST_CROSS_RADIUS = 0.090
    LOOSE_CROSS_RADIUS = 0.125
    ELITE_BLEND_COUNT = 1
    ELITE_TEMPERATURE = 18.0

    TRACK_LOOKAHEAD = 0.060

    # ========== debug / diagnostics ==========
    # Single switch for all per-step diagnostic prints. Per-episode summaries
    # (gate-discovery, episode end) print regardless so a baseline log is
    # always informative without flooding the output.
    DEBUG_PRINT_ENABLED = True
    DEBUG_EVERY_PLANS = 10

    FORGET_OLD_SCENARIOS = True
    USE_PREVIOUS_ROUTE_CANDIDATE = False
    RESET_SIGMA_EACH_PLAN = True
    COMMAND_ONLY_NONCRASHING = True

    # 4-D sigma: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    # Original accel sigma [2.8, 2.8, 1.8] m/s² converts to:
    #   roll/pitch ≈ a/g ≈ 0.29 rad,  thrust ≈ a*m/cmd_f ≈ 0.081
    SIGMA_INIT = np.array([0.35, 0.35, 0.25, 0.080], dtype=np.float64)
    SIGMA_MIN = np.array([0.06, 0.06, 0.04, 0.015], dtype=np.float64)
    SIGMA_MAX = np.array([0.60, 0.60, 0.45, 0.150], dtype=np.float64)

    # Route layer
    ROUTE_PRESELECT_K = 4
    ROUTE_TOP_K = 2
    ROUTE_MEAN_BLEND = 1.00
    ROUTE_COMMAND_BLEND = 0.24
    ROUTE_SAMPLE_SPACING = 0.18
    ROUTE_OBS_LOOKAHEAD_MARGIN = 0.55
    ROUTE_SIDE_OFFSETS = (0.0, 0.35, -0.35, 0.65, -0.65)
    ROUTE_ENTRY_Y_OFFSETS = (0.0, 0.08, -0.08)
    ROUTE_ENTRY_Z_OFFSETS = (0.0, 0.06, -0.06)

    # ========== cost weights ==========
    W_REF_POS = 11.0
    W_REF_VEL = 3.0
    W_TERMINAL_REF = 40.0
    W_PROGRESS = 26.0
    W_GATE_DISTANCE_STAGE = 5.5
    W_GATE_DISTANCE_TERMINAL = 42.0
    W_GATE_CLOSING = 28.0
    W_NOT_CROSSED_REACHABLE = 1800.0
    W_CROSS_CENTER = 4500.0        # strongly reward center crossings (was 1900)
    W_BAD_CROSS = 30000.0
    BONUS_GOOD_CROSS = 2100.0

    W_POLE_BUFFER = 2600.0
    W_POLE_COLLISION = 25000.0
    W_POLE_NEAR_EXP = 15.0

    W_FRAME_BUFFER = 9000.0         # stronger soft buffer penalty (was 6200)
    W_FRAME_COLLISION = 80000.0     # stronger collision penalty (was 56000)
    W_FRAME_SLAB = 35000.0          # stronger slab penalty (was 26000)
    W_FUNNEL = 2800.0               # much stronger funnel centering (was 900)

    W_ALTITUDE = 6000.0
    W_ALTITUDE_HARD = 40000.0
    W_SPEED_LIMIT = 18.0
    W_INPUT = 0.025
    W_DINPUT = 0.075
    W_LATERAL = 0.020

    # --- NEW spatial / rotational cost weights ---
    W_CORRIDOR = 800.0          # soft penalty for leaving the spatial corridor
    W_CORRIDOR_HARD = 4000.0    # hard penalty for deep corridor violation
    W_SPATIAL_PROGRESS = 4.0    # reward longitudinal progress ds
    W_CURVATURE_SPEED = 2.0     # track curvature-adapted speed reference
    W_ATTITUDE_SMOOTH = 0.15    # penalise large attitude rates
    W_ATTITUDE_LIMIT = 400.0    # penalise roll/pitch beyond safe limit

    # Projection-based route tracking
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

    # APF reactive push
    APF_INFLUENCE = 0.36
    APF_GAIN = 0.22
    APF_MAX = 0.65

    # Launch altitude hold
    LAUNCH_HOLD_TIME = 0.25
    LAUNCH_BLEND_TIME = 0.70
    VZ_REF_MAX = 0.40

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(self, obs: dict[str, "NDArray[np.floating]"], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._g = 9.81
        self._dt = 1.0 / float(config.env.freq)

        # --- drone physical parameters ---
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(drone_params["mass"])
        self._thrust_min = float(drone_params["thrust_min"]) * 4.0
        self._thrust_max = float(drone_params["thrust_max"]) * 4.0
        self._mass_estimate = self._mass
        self._physics = config.sim.physics

        # Rotational dynamics coefficients (same across so_rpy variants)
        self._rpy_coef = np.array(drone_params["rpy_coef"], dtype=np.float64)
        self._rpy_rates_coef = np.array(drone_params["rpy_rates_coef"], dtype=np.float64)
        self._cmd_rpy_coef = np.array(drone_params["cmd_rpy_coef"], dtype=np.float64)
        self._acc_coef = float(drone_params.get("acc_coef", 0.0))
        self._cmd_f_coef = float(drone_params.get("cmd_f_coef", 0.96836458))

        # Rotor lag (so_rpy_rotor / so_rpy_rotor_drag)
        self._has_rotor_lag = "thrust_time_coef" in drone_params
        self._thrust_time_coef = float(drone_params.get("thrust_time_coef", 1.0))

        # Aerodynamic drag (so_rpy_rotor_drag only)
        self._has_drag = "drag_linear_coef" in drone_params
        self._drag_xy = float(drone_params.get("drag_linear_coef", 0.0))
        self._drag_z = float(drone_params.get("drag_square_coef", 0.0))

        # Hover thrust command (simulator applies cmd_f_coef internally)
        self._hover_thrust = self._mass * self._g / self._cmd_f_coef

        # Thrust-lag state (actual thrust after rotor response)
        self._thrust_actual = self._hover_thrust

        # Hover control vector: [roll=0, pitch=0, yaw=0, thrust=hover]
        self._hover_u = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)

        # --- gate / obstacle setup ---
        self._gate_positions = np.array(
            [g.tolist() for g in obs["gates_pos"]], dtype=np.float64,
        )
        self._gate_quats = np.array(
            [g.tolist() for g in obs["gates_quat"]], dtype=np.float64,
        )
        self._gate_rotmats = [Rot.from_quat(q).as_matrix() for q in self._gate_quats]
        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])

        self._obstacle_positions = np.array(
            [p.tolist() for p in obs["obstacles_pos"]], dtype=np.float64,
        )
        self._n_obstacles = len(self._obstacle_positions)
        self._gates_visited = obs["gates_visited"].copy()

        # Index of the most recently passed gate, or -1 if none yet. Tracked
        # so its frame stays in the MPPI cost while the drone exits and
        # turns toward the next gate (prevents post-pass exit clips).
        self._prev_target_gate = -1
        print(f"[SPATIAL-INIT] drone_pos={obs['pos']}")
        for i in range(self._n_gates):
            print(f"SPATIAL-INIT] gate{i}: pos={self._gate_positions[i]} visited={self._gates_visited[i]}")
        self._obstacles_visited = np.array(
            obs.get("obstacles_visited", np.zeros(len(self._obstacle_positions), dtype=bool)),
            dtype=bool,
        )

        # --- spatial geometry engine (Bishop frame + corridors) ---
        self._geo = None
        self._prev_s = 0.0
        if _HAS_SPATIAL:
            self._init_geometry_engine(obs, info, config)

        # --- MPPI sampling distribution (4-D) ---
        seed_value = getattr(config, "seed", None)
        seed = int(seed_value) if seed_value is not None else None
        self._rng = np.random.default_rng(seed)

        self._mean_u = np.tile(self._hover_u, (self.MPC_N, 1))
        horizon_ramp = np.linspace(0.85, 1.20, self.MPC_N)[:, None]
        self._base_sigma = horizon_ramp * self.SIGMA_INIT[None, :]
        self._sigma = self._base_sigma.copy()

        # --- planner timing ---
        self._planner_interval_steps = max(
            1, int(round(1.0 / max(self.PLANNER_HZ * self._dt, _EPS))),
        )
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

        self._prev_u = self._hover_u.copy()
        self._prev_output = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)
        self._last_pos = np.zeros(3, dtype=np.float64)
        self._last_rpy = np.zeros(3, dtype=np.float64)
        self._last_drpy = np.zeros(3, dtype=np.float64)
        self._last_best_traj = None
        self._last_mean_traj = None
        self._active_route_points = None
        self._route_candidates: list[np.ndarray] = []
        self._route_costs: list[float] = []
        self._last_route_cost = math.inf
        self._last_route_anchor_cost = math.inf
        self._last_route_anchor_margin = math.inf
        self._last_route_anchor_crossed = False
        self._last_route_anchor_traj = None
        self._warped_discovery_seq = None  # smooth-warped trajectory on object discovery
        self._last_selected_idx = -1
        self._last_best_cost = math.inf
        self._last_best_margin = math.inf
        self._launch_z: float | None = None
        self._tick = 0
        self._finished = False
        self._last_plan_ms = 0.0
        self._plan_ms_ema = 0.0
        self._plan_counter = 0
        self._summary_printed = False

        print(
            f"[SPATIAL-SCENARIO-MPC] ready: K={self.K_SAMPLES}, N={self.MPC_N}, "
            f"dt={self.MPC_DT:.3f}s, horizon={self.MPC_N * self.MPC_DT:.2f}s, "
            f"planner_hz={self.PLANNER_HZ:.1f}, "
            f"physics={self._physics}, "
            f"rotor_lag={'YES' if self._has_rotor_lag else 'NO'}, "
            f"drag={'YES' if self._has_drag else 'NO'}, "
            f"spatial={'YES' if self._geo is not None else 'NO'}"
        )

    # ------------------------------------------------------------------
    #  Spatial geometry initialisation
    # ------------------------------------------------------------------
    def _init_geometry_engine(self, obs, info, config):
        """Build the GeometryEngine for Bishop-frame spatial coordinates."""
        self._rebuild_geometry()

    def _rebuild_geometry(self):
        """(Re)build GeometryEngine from current gate/obstacle positions."""
        if not _HAS_SPATIAL:
            return
        gates_pos = [g.tolist() for g in self._gate_positions]
        rots = Rot.from_quat(self._gate_quats)
        mats = rots.as_matrix()
        gates_normal = mats[:, :, 0]
        gates_y = mats[:, :, 1]
        gates_z = mats[:, :, 2]
        obstacles_pos = [p.tolist() for p in self._obstacle_positions]
        try:
            self._geo = GeometryEngine(
                gates_pos=gates_pos,
                gates_normal=gates_normal,
                gates_y=gates_y,
                gates_z=gates_z,
                obstacles_pos=obstacles_pos,
            )
            self._prev_s = 0.0
            print(f"[SPATIAL] GeometryEngine built, path_length={self._geo.total_length:.2f}m")
        except Exception as e:
            import traceback
            print(f"[SPATIAL] GeometryEngine build failed: {e}")
            traceback.print_exc()
            self._geo = None

    # ------------------------------------------------------------------
    #  Cartesian ↔ Spatial conversions
    # ------------------------------------------------------------------
    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        """Convert Cartesian state to spatial curvilinear state.

        Returns (s, w1, w2, ds, dw1, dw2) using the Bishop frame.
        """
        if self._geo is None:
            return None

        s = self._geo.get_closest_s(pos, s_guess=self._prev_s)
        self._prev_s = s

        f = self._geo.get_frame(s)
        r_vec = pos - f["pos"]
        w1 = float(np.dot(r_vec, f["n1"]))
        w2 = float(np.dot(r_vec, f["n2"]))

        h = max(1.0 - f["k1"] * w1 - f["k2"] * w2, 0.01)
        ds = float(np.dot(vel, f["t"])) / h
        dw1 = float(np.dot(vel, f["n1"]))
        dw2 = float(np.dot(vel, f["n2"]))

        return np.array([s, w1, w2, ds, dw1, dw2], dtype=np.float64)

    def _spatial_corridor_cost_batch(self, positions: np.ndarray) -> np.ndarray:
        """Compute soft corridor violation cost for a batch (K,3) of positions.

        Projects each position onto the spatial frame and penalises lateral
        deviation w1 that is outside the precomputed corridor bounds.
        Samples a sparse subset for speed.
        """
        if self._geo is None:
            return np.zeros(positions.shape[0], dtype=np.float64)

        K = positions.shape[0]
        cost = np.zeros(K, dtype=np.float64)

        # Sample every 4th point for speed (corridor is spatially smooth)
        stride = min(4, max(1, K // 10))
        for i in range(0, K, stride):
            s = self._geo.get_closest_s(positions[i], s_guess=self._prev_s)
            f = self._geo.get_frame(s)
            r_vec = positions[i] - f["pos"]
            w1 = float(np.dot(r_vec, f["n1"]))

            lb, ub = self._geo.get_static_bounds(s)

            viol_lo = max(lb - w1, 0.0)
            viol_hi = max(w1 - ub, 0.0)
            viol = viol_lo + viol_hi
            c = self.W_CORRIDOR * viol ** 2

            deep = max(viol - 0.10, 0.0)
            c += self.W_CORRIDOR_HARD * deep ** 2

            # Apply to this sample and its neighbors
            end = min(i + stride, K)
            cost[i:end] = c

        return cost

    def _spatial_progress_and_curvature(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-sample (progress_reward, curvature_speed_cost).

        Uses a single shared frame lookup (centroid) for speed.
        """
        if self._geo is None:
            return np.zeros(positions.shape[0]), np.zeros(positions.shape[0])

        K = positions.shape[0]
        # Use centroid position for a single frame lookup
        centroid = np.mean(positions, axis=0)
        s = self._geo.get_closest_s(centroid, s_guess=self._prev_s)
        f = self._geo.get_frame(s)
        t_vec = f["t"]

        # Vectorised progress: dot(vel, tangent)
        progress = np.maximum(velocities @ t_vec, 0.0)

        # Curvature-based speed limit
        k_mag = math.sqrt(f["k1"] ** 2 + f["k2"] ** 2)
        v_corner = math.sqrt(5.0 / (k_mag + 0.01))
        speeds = np.linalg.norm(velocities, axis=1)
        overspeed = np.maximum(speeds - min(v_corner, self.V_MAX), 0.0)
        curv_cost = overspeed ** 2

        return progress, curv_cost

    # ------------------------------------------------------------------
    #  Full RPY + thrust dynamics rollout (vectorised)
    # ------------------------------------------------------------------
    # Number of sub-steps per MPC interval for numerical stability.
    # The RPY dynamics (eigenvalue magnitude ~13.7 rad/s) require dt < 0.068s
    # for explicit Euler stability.  With MPC_DT=0.11s, 2 sub-steps gives
    # sub-dt = 0.055s which is within the stability bound (|1+dt*λ| ≈ 0.92).
    ROLLOUT_SUB_STEPS = 2

    def _rollout_rpy_thrust(
        self,
        samples: np.ndarray,
        pos0: np.ndarray,
        vel0: np.ndarray,
        rpy0: np.ndarray,
        drpy0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised forward integration using the full rotational dynamics.

        Supports three physics levels:
          - so_rpy: instant thrust, no drag
          - so_rpy_rotor: first-order thrust lag
          - so_rpy_rotor_drag: thrust lag + body-frame linear drag

        Args:
            samples: (K, N, 4) control sequences [phi_c, theta_c, psi_c, T_c]
            pos0, vel0, rpy0, drpy0: initial state (3,) each

        Returns:
            all_pos:  (K, N+1, 3)
            all_vel:  (K, N+1, 3)
            all_rpy:  (K, N+1, 3)
            all_drpy: (K, N+1, 3)
        """
        K, N, _ = samples.shape
        n_sub = self.ROLLOUT_SUB_STEPS
        sub_dt = self.MPC_DT / n_sub

        pos = np.tile(pos0, (K, 1)).astype(np.float64)
        vel = np.tile(vel0, (K, 1)).astype(np.float64)
        rpy = np.tile(rpy0, (K, 1)).astype(np.float64)
        drpy = np.tile(drpy0, (K, 1)).astype(np.float64)

        all_pos = np.empty((K, N + 1, 3), dtype=np.float64)
        all_vel = np.empty((K, N + 1, 3), dtype=np.float64)
        all_rpy = np.empty((K, N + 1, 3), dtype=np.float64)
        all_drpy = np.empty((K, N + 1, 3), dtype=np.float64)

        all_pos[:, 0] = pos
        all_vel[:, 0] = vel
        all_rpy[:, 0] = rpy
        all_drpy[:, 0] = drpy

        c_rpy = self._rpy_coef       # (3,)
        c_drpy = self._rpy_rates_coef  # (3,)
        c_cmd = self._cmd_rpy_coef    # (3,)
        g_z = -self._g
        inv_mass = 1.0 / self._mass_estimate
        acc_coef = self._acc_coef
        cmd_f = self._cmd_f_coef

        # Rotor lag state
        has_rotor_lag = self._has_rotor_lag
        if has_rotor_lag:
            T_act = np.full((K, 1), self._thrust_actual, dtype=np.float64)
            sub_dt_inv_tau = sub_dt / self._thrust_time_coef

        # Drag coefficients
        has_drag = self._has_drag
        if has_drag:
            drag_xy = self._drag_xy
            drag_z_coef = self._drag_z

        for k in range(N):
            u = samples[:, k, :]        # (K, 4)
            cmd_rpy_k = u[:, :3]        # (K, 3) roll/pitch/yaw commands
            T_cmd = u[:, 3:4]           # (K, 1) thrust command

            for _ in range(n_sub):
                # --- Rotor lag: first-order thrust response ---
                if has_rotor_lag:
                    T_act = T_act + (T_cmd - T_act) * sub_dt_inv_tau
                    T_phys = acc_coef + cmd_f * T_act
                else:
                    T_phys = acc_coef + cmd_f * T_cmd

                # --- Rotational dynamics (sub-step) ---
                ddrpy = c_rpy[None, :] * rpy + c_drpy[None, :] * drpy + c_cmd[None, :] * cmd_rpy_k

                # --- Rotation matrix elements ---
                phi   = rpy[:, 0:1]
                theta = rpy[:, 1:2]
                psi   = rpy[:, 2:3]

                cx, sx = np.cos(phi), np.sin(phi)
                cy, sy = np.cos(theta), np.sin(theta)
                cz, sz = np.cos(psi), np.sin(psi)

                # Third column of R_IB (body Z in world) — for thrust
                R_02 = sx * sz + cx * cz * sy   # (K, 1)
                R_12 = cx * sy * sz - cz * sx   # (K, 1)
                R_22 = cx * cy                   # (K, 1)

                # --- World-frame acceleration ---
                acc = np.empty_like(vel)
                acc[:, 0:1] = R_02 * T_phys * inv_mass
                acc[:, 1:2] = R_12 * T_phys * inv_mass
                acc[:, 2:3] = g_z + R_22 * T_phys * inv_mass

                # --- Aerodynamic drag (body-frame linear) ---
                if has_drag:
                    # Full rotation matrix columns 0,1 for drag transform
                    R_00 = cz * cy
                    R_01 = cz * sy * sx - sz * cx
                    R_10 = sz * cy
                    R_11 = sz * sy * sx + cz * cx
                    R_20 = -sy
                    R_21 = cy * sx
                    # World velocity → body frame: v_body = R^T @ v_world
                    vx, vy, vz_w = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]
                    vb_x = R_00 * vx + R_10 * vy + R_20 * vz_w
                    vb_y = R_01 * vx + R_11 * vy + R_21 * vz_w
                    vb_z = R_02 * vx + R_12 * vy + R_22 * vz_w
                    # Drag force in body frame (coefficients are negative → opposes motion)
                    fd_x = drag_xy * vb_x
                    fd_y = drag_xy * vb_y
                    fd_z = drag_z_coef * vb_z
                    # Transform back to world: a_drag = R @ F_body / m
                    acc[:, 0:1] += (R_00 * fd_x + R_01 * fd_y + R_02 * fd_z) * inv_mass
                    acc[:, 1:2] += (R_10 * fd_x + R_11 * fd_y + R_12 * fd_z) * inv_mass
                    acc[:, 2:3] += (R_20 * fd_x + R_21 * fd_y + R_22 * fd_z) * inv_mass

                # --- Semi-implicit Euler (sub-step) ---
                drpy = drpy + ddrpy * sub_dt
                rpy = rpy + drpy * sub_dt

                vel = vel + acc * sub_dt
                pos = pos + vel * sub_dt

            all_pos[:, k + 1] = pos
            all_vel[:, k + 1] = vel
            all_rpy[:, k + 1] = rpy
            all_drpy[:, k + 1] = drpy

        return all_pos, all_vel, all_rpy, all_drpy

    # ------------------------------------------------------------------
    #  Geometry helpers (from original scenario MPC)
    # ------------------------------------------------------------------
    def _get_gate_normal(self, gi: int, from_pos: np.ndarray) -> np.ndarray:
        if gi < 0 or gi >= self._n_gates:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        normal = self._gate_rotmats[gi][:, 0].copy()
        to_gate = self._gate_positions[gi] - from_pos
        if float(np.dot(to_gate, normal)) < 0.0:
            normal = -normal
        n = np.linalg.norm(normal)
        return normal / max(float(n), _EPS)

    @staticmethod
    def _unit(v, fallback=None):
        n = float(np.linalg.norm(v))
        if n > _EPS:
            return np.asarray(v, dtype=np.float64) / n
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).copy()

    @staticmethod
    def _world_to_gate_local(points, gp, R):
        return (points - gp) @ R

    @staticmethod
    def _point_segment_distance_xy(point_xy, a_xy, b_xy):
        ab = b_xy - a_xy
        denom = float(np.dot(ab, ab))
        if denom < _EPS:
            return float(np.linalg.norm(point_xy - a_xy)), 0.0, a_xy.copy()
        t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
        closest = a_xy + t * ab
        return float(np.linalg.norm(point_xy - closest)), t, closest

    def _apply_launch_altitude_ramp(self, ref_pos, ref_vel):
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

    def _distance_to_gate_bars(self, local):
        """Distance to the nearest gate collision box (matching MuJoCo gate.xml).

        The gate frame consists of 4 collision boxes (from gate.xml):
          top:    center (0, 0, +0.28), half-extents (0.01, 0.36, 0.08)
          bottom: center (0, 0, -0.28), half-extents (0.01, 0.36, 0.08)
          left:   center (0, -0.28, 0), half-extents (0.01, 0.08, 0.36)
          right:  center (0, +0.28, 0), half-extents (0.01, 0.08, 0.36)

        Returns the minimum unsigned distance from each point to any box surface.
        """
        K = local.shape[0]
        x = local[:, 0]
        y = local[:, 1]
        z = local[:, 2]

        # Box definitions: (center_y, center_z, half_y, half_z)
        # All boxes have half_x = 0.01 (gate plane thickness)
        hx = 0.01
        boxes = [
            (0.0,   0.28, 0.36, 0.08),   # top
            (0.0,  -0.28, 0.36, 0.08),   # bottom
            (-0.28,  0.0, 0.08, 0.36),   # left
            (0.28,   0.0, 0.08, 0.36),   # right
        ]

        d_min = np.full(K, 1e6, dtype=np.float64)
        for cy, cz, hy, hz in boxes:
            # Signed distance components for an axis-aligned box
            dx = np.abs(x) - hx
            dy = np.abs(y - cy) - hy
            dz = np.abs(z - cz) - hz
            # Outside distance: Euclidean of positive components
            outside = np.sqrt(
                np.maximum(dx, 0.0) ** 2
                + np.maximum(dy, 0.0) ** 2
                + np.maximum(dz, 0.0) ** 2,
            )
            # Inside distance: max of negative components (closest wall)
            inside = np.maximum(dx, np.maximum(dy, dz))
            # Unsigned distance: 0 when inside, positive when outside
            dist = np.where(inside < 0.0, 0.0, outside)
            d_min = np.minimum(d_min, dist)

        return d_min

    def _gate_geometry_cost_and_margin(self, pos, gp, R, active_funnel):
        local = self._world_to_gate_local(pos, gp, R)
        x_abs = np.abs(local[:, 0])
        y_abs = np.abs(local[:, 1])
        z_abs = np.abs(local[:, 2])
        cost = np.zeros(pos.shape[0], dtype=np.float64)

        d_bar = self._distance_to_gate_bars(local)
        # Inflate distances by drone body radius so the frame check accounts
        # for the physical extent of the Crazyflie, not just its center point.
        d_bar_eff = d_bar - self.DRONE_RADIUS
        # d_bar_eff <= 0 means the drone body overlaps a gate frame box
        frame_soft = self.GATE_FRAME_BUFFER  # start penalising at this distance from surface
        v_soft = np.maximum(frame_soft - d_bar_eff, 0.0)
        v_hard = np.maximum(-d_bar_eff, 0.0)  # inside the box
        cost += self.W_FRAME_BUFFER * (v_soft / max(frame_soft, _EPS)) ** 2
        cost += self.W_FRAME_COLLISION * (v_hard / max(self.DRONE_RADIUS, _EPS)) ** 2

        in_slab = x_abs < self.GATE_PLANE_SLAB
        # Shrink the safe opening by the drone radius
        effective_half_open = max(self.GATE_HALF_OPENING - self.DRONE_RADIUS, 0.02)
        outside_open = np.maximum(y_abs - effective_half_open, z_abs - effective_half_open)
        outside_open = np.maximum(outside_open, 0.0)
        slab_scale = np.maximum(1.0 - x_abs / self.GATE_PLANE_SLAB, 0.0)
        cost += self.W_FRAME_SLAB * in_slab * slab_scale * (outside_open / effective_half_open) ** 2

        # Funnel safe boundary: at gate plane, allow ±0.10m (5cm buffer from
        # actual collision at ±0.15m).  Widens linearly to FUNNEL_OUTER_HALF
        # at FUNNEL_LENGTH away from the gate.
        gate_safe = max(self.GATE_HALF_OPENING - self.DRONE_RADIUS - 0.05, 0.04)  # ~0.10m

        if active_funnel:
            alpha = np.clip(x_abs / self.FUNNEL_LENGTH, 0.0, 1.0)
            h_bound = gate_safe + alpha * (self.FUNNEL_OUTER_HALF - gate_safe)
            h_bound = np.maximum(h_bound, 0.02)
            v_y = np.maximum(y_abs - h_bound, 0.0)
            v_z = np.maximum(z_abs - h_bound, 0.0)
            fw = 1.0 - 0.55 * alpha
            cost += self.W_FUNNEL * fw * ((v_y / h_bound) ** 2 + (v_z / h_bound) ** 2)
        else:
            # Exit-side funnel: still penalise being outside the opening on
            # the exit side so sharp post-gate turns don't clip the frame.
            exit_alpha = np.clip(x_abs / self.EXIT_DIST, 0.0, 1.0)
            exit_bound = gate_safe + exit_alpha * (self.FUNNEL_OUTER_HALF - gate_safe)
            exit_bound = np.maximum(exit_bound, 0.02)
            v_ey = np.maximum(y_abs - exit_bound, 0.0)
            v_ez = np.maximum(z_abs - exit_bound, 0.0)
            exit_fw = 1.0 - 0.50 * exit_alpha
            cost += 0.80 * self.W_FUNNEL * exit_fw * ((v_ey / exit_bound) ** 2 + (v_ez / exit_bound) ** 2)

        margin_bar = d_bar_eff  # positive = clear, negative = inside box
        return cost, margin_bar

    # ------------------------------------------------------------------
    #  4-D control utilities
    # ------------------------------------------------------------------
    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        """Clip a 4-D control vector or batch [roll_cmd, pitch_cmd, yaw_cmd, thrust]."""
        out = np.array(u, dtype=np.float64, copy=True)
        out[..., 0] = np.clip(out[..., 0], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
        out[..., 1] = np.clip(out[..., 1], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
        out[..., 2] = np.clip(out[..., 2], -self.MAX_YAW_CMD, self.MAX_YAW_CMD)
        out[..., 3] = np.clip(out[..., 3], self._thrust_min, self._thrust_max)
        return out

    # ------------------------------------------------------------------
    #  Deterministic route candidates (Cartesian — reused from scenario MPC)
    # ------------------------------------------------------------------
    def _clip_route_point(self, point):
        p = np.asarray(point, dtype=np.float64).copy()
        p[2] = float(np.clip(p[2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05))
        return p

    def _clean_route(self, points):
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

    def _sample_polyline(self, route, spacing=None):
        if spacing is None:
            spacing = self.ROUTE_SAMPLE_SPACING
        route = np.asarray(route, dtype=np.float64)
        if len(route) <= 1:
            return route.copy()
        pts = [route[0].copy()]
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            d = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(d / max(spacing, 0.03))))
            for j in range(1, n + 1): 
                pts.append((1.0 - j / n) * a + (j / n) * b)
        return np.vstack(pts)

    # ------------------------------------------------------------------
    #  Route deflection: push route segments away from gate frames
    # ------------------------------------------------------------------
    ROUTE_GATE_SAFE_DIST = 0.18   # min distance from route to any gate frame bar
    ROUTE_GATE_CHECK_SPACING = 0.12  # sample spacing for gate proximity check (was 0.08)

    def _min_dist_to_gate_frame(self, point, gate_idx):
        """Distance from a single world-space point to gate frame bars."""
        gp = self._gate_positions[gate_idx]
        R = self._gate_rotmats[gate_idx]
        local = (point - gp) @ R  # (3,)
        local_2d = local.reshape(1, 3)
        return float(self._distance_to_gate_bars(local_2d)[0])

    def _gate_frame_push_vector(self, point, gate_idx):
        """Compute a push vector pointing away from the nearest gate frame bar.

        Returns (push_direction, distance_to_nearest_bar).
        push_direction is a unit vector in world space pointing away from the bar.
        """
        gp = self._gate_positions[gate_idx]
        R = self._gate_rotmats[gate_idx]
        local = (point - gp) @ R  # (3,)

        # Find nearest bar center in gate-local coords
        hx = 0.01
        boxes = [
            (0.0,   0.28, 0.36, 0.08),   # top
            (0.0,  -0.28, 0.36, 0.08),   # bottom
            (-0.28,  0.0, 0.08, 0.36),   # left
            (0.28,   0.0, 0.08, 0.36),   # right
        ]
        best_dist = 1e6
        best_push_local = np.array([0.0, 0.0, 0.0])
        for cy, cz, hy, hz in boxes:
            dx = abs(local[0]) - hx
            dy = abs(local[1] - cy) - hy
            dz = abs(local[2] - cz) - hz
            outside = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2 + max(dz, 0.0)**2)
            inside = max(dx, max(dy, dz))
            dist = 0.0 if inside < 0.0 else outside
            if dist < best_dist:
                best_dist = dist
                # Push direction: from box center toward point, in gate-local frame
                push = np.array([
                    local[0],
                    local[1] - cy,
                    local[2] - cz,
                ], dtype=np.float64)
                norm = float(np.linalg.norm(push))
                if norm > _EPS:
                    best_push_local = push / norm
                else:
                    best_push_local = np.array([0.0, -np.sign(cy + _EPS), -np.sign(cz + _EPS)])

        # Convert push direction back to world frame
        push_world = R @ best_push_local
        return push_world, best_dist

    def _batch_min_dist_to_gates(self, points):
        """Vectorized: min distance from each point to any gate frame bar across all gates.

        Returns (min_dists, nearest_gate_idx) arrays of shape (N,).
        Only checks gates within a coarse bounding-sphere pre-filter.
        """
        N = len(points)
        min_dists = np.full(N, 1e6, dtype=np.float64)
        nearest_gi = np.zeros(N, dtype=np.int32)
        for gi in range(self._n_gates):
            gp = self._gate_positions[gi]
            # Coarse pre-filter: skip gates whose center is > 1.2m away
            coarse_d = np.linalg.norm(points - gp, axis=1)
            close_mask = coarse_d < 1.2
            if not np.any(close_mask):
                continue
            R = self._gate_rotmats[gi]
            local = self._world_to_gate_local(points[close_mask], gp, R)
            dists = self._distance_to_gate_bars(local)
            update = dists < min_dists[close_mask]
            idx = np.where(close_mask)[0]
            min_dists[idx[update]] = dists[update]
            nearest_gi[idx[update]] = gi
        return min_dists, nearest_gi

    def _deflect_route_from_gates(self, route):
        """Insert intermediate waypoints where route segments pass too close to gate frames.

        Uses vectorized batch distance checks for speed.
        """
        if self._n_gates == 0 or len(route) < 2:
            return route

        safe_dist = self.ROUTE_GATE_SAFE_DIST + self.DRONE_RADIUS

        # Sample the whole route at once
        sampled = self._sample_polyline(route, spacing=self.ROUTE_GATE_CHECK_SPACING)
        if len(sampled) < 2:
            return route

        # Batch distance check against all gates
        dists, nearest_gi = self._batch_min_dist_to_gates(sampled)

        # For the target gate, mask out points near the gate crossing zone
        if 0 <= self._target_gate < self._n_gates:
            gp_t = self._gate_positions[self._target_gate]
            normal_t = self._get_gate_normal(self._target_gate, route[0])
            signed_t = (sampled - gp_t) @ normal_t
            # Don't deflect near target gate crossing zone (|signed| < 0.30)
            target_crossing = (nearest_gi == self._target_gate) & (np.abs(signed_t) <= 0.30)
            dists[target_crossing] = 1e6  # mark as safe

        # Find violations
        violation_mask = dists < safe_dist
        if not np.any(violation_mask):
            return route  # fast path: no deflection needed

        # For each violating point, compute push and insert deflected waypoints
        viol_idx = np.where(violation_mask)[0]
        new_points = [route[0].copy()]

        # Map sampled indices back to route segments
        seg_starts = [0]
        cum = 0
        for i in range(len(route) - 1):
            seg_len = float(np.linalg.norm(route[i + 1] - route[i]))
            n_pts = max(1, int(math.ceil(seg_len / self.ROUTE_GATE_CHECK_SPACING)))
            cum += n_pts
            seg_starts.append(cum)

        for seg_i in range(len(route) - 1):
            s_start, s_end = seg_starts[seg_i], seg_starts[seg_i + 1]
            seg_viols = viol_idx[(viol_idx >= s_start) & (viol_idx < s_end)]

            if len(seg_viols) > 0:
                # Quantize to at most 3 deflection points per segment
                inserted = set()
                for vi in seg_viols:
                    n_in_seg = max(s_end - s_start, 1)
                    t_param = (vi - s_start) / n_in_seg
                    t_bucket = round(t_param * 3) / 3
                    if t_bucket in inserted or t_bucket <= 0.0 or t_bucket >= 1.0:
                        continue
                    inserted.add(t_bucket)
                    pt = sampled[vi]
                    gi = int(nearest_gi[vi])
                    push, _ = self._gate_frame_push_vector(pt, gi)
                    push_mag = max(safe_dist - dists[vi] + 0.06, 0.04)
                    deflected = pt + push_mag * push
                    deflected[2] = float(np.clip(
                        deflected[2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05,
                    ))
                    new_points.append(deflected)

            new_points.append(route[seg_i + 1].copy())

        return self._clean_route(new_points)

    def _straighten_near_gates(self, route):
        """Clamp route waypoints near the target gate to the gate's normal axis.

        When the drone is close to a gate (within FUNNEL_LENGTH), the route
        should approach/exit on a perfectly straight line along the gate
        normal.  This prevents lateral drift during dips and tight crossings.
        Only affects the target gate, and skips the first waypoint (drone pos).
        """
        if self._n_gates == 0 or len(route) < 2:
            return route
        if not (0 <= self._target_gate < self._n_gates):
            return route

        out = route.copy()
        gi = self._target_gate
        gp = self._gate_positions[gi]
        normal = self._get_gate_normal(gi, route[0])
        R = self._gate_rotmats[gi]
        gate_y = R[:, 1].copy()
        gate_z = R[:, 2].copy()
        proximity = self.FUNNEL_LENGTH + 0.10  # straighten zone
        # Max allowed offset in gate-local Y/Z to stay clear of frame bars
        max_local_offset = self.GATE_HALF_OPENING - self.DRONE_RADIUS - 0.03  # ~0.12

        # Skip i=0 (drone's current position — don't move it)
        for i in range(1, len(out)):
            pt = out[i]
            to_pt = pt - gp
            along = float(np.dot(to_pt, normal))
            if abs(along) > proximity:
                continue
            lateral = to_pt - along * normal
            lat_mag = float(np.linalg.norm(lateral))

            blend = 1.0 - abs(along) / proximity
            blend = max(0.0, min(blend, 1.0)) ** 1.5

            if lat_mag >= 0.02:
                out[i] = gp + along * normal + (1.0 - blend) * lateral

            # Clamp gate-local Y/Z to keep route within the safe opening
            # The allowed offset relaxes toward the funnel edge
            clamped_max = max_local_offset + (1.0 - blend) * 0.15
            residual = out[i] - (gp + along * normal)
            local_y = float(np.dot(residual, gate_y))
            local_z = float(np.dot(residual, gate_z))
            local_y_c = float(np.clip(local_y, -clamped_max, clamped_max))
            local_z_c = float(np.clip(local_z, -clamped_max, clamped_max))
            if local_y != local_y_c or local_z != local_z_c:
                out[i] = gp + along * normal + local_y_c * gate_y + local_z_c * gate_z

            out[i, 2] = float(np.clip(out[i, 2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05))

        return out

    def _route_min_gate_clearance(self, route):
        """Return the minimum distance from sampled route points to any gate frame bar."""
        if self._n_gates == 0 or len(route) < 2:
            return 1.0
        sampled = self._sample_polyline(route, spacing=self.ROUTE_GATE_CHECK_SPACING)
        dists, nearest_gi = self._batch_min_dist_to_gates(sampled)
        dists = dists - self.DRONE_RADIUS

        if 0 <= self._target_gate < self._n_gates:
            gp_t = self._gate_positions[self._target_gate]
            normal_t = self._get_gate_normal(self._target_gate, route[0])
            signed_t = (sampled - gp_t) @ normal_t
            target_crossing = (nearest_gi == self._target_gate) & (np.abs(signed_t) <= 0.25)
            dists[target_crossing] = 1e6

        return float(np.min(dists)) if len(dists) > 0 else 1.0

    def _make_gate_tail(self, pos):
        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            return [pos.copy()]
        gp = self._gate_positions[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)
        approach = gp - self.APPROACH_DIST * normal
        entry = gp - self.FUNNEL_LENGTH * normal
        exit_pt = gp + self.EXIT_DIST * normal
        signed = float(np.dot(pos - gp, normal))
        tail = []
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

    def _score_route(self, route, pos, vel=None):
        route = self._clean_route(route)
        if len(route) <= 1:
            return 1e9
        diffs = route[1:] - route[:-1]
        seg_lengths = np.linalg.norm(diffs, axis=1)
        length = float(np.sum(seg_lengths))
        score = 1.00 * length

        first_dir = self._unit(route[1] - route[0])

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            gate_dir = self._unit(gp - pos, fallback=first_dir)
            align = float(np.dot(first_dir, gate_dir))
            score += 7.0 * max(0.0, 1.0 - align)
            cross_track, _, _ = self._point_segment_distance_xy(route[1, :2], pos[:2], gp[:2])
            score += 26.0 * cross_track * cross_track

        # Penalise routes whose initial direction opposes the drone's current
        # velocity.  This prevents MPPI from wasting samples on trajectories
        # that would require an instantaneous >90-degree turn.
        if vel is not None:
            speed_xy = float(np.linalg.norm(vel[:2]))
            if speed_xy > 0.35:
                vel_dir = self._unit(np.array([vel[0], vel[1], 0.0]))
                first_xy = self._unit(np.array([first_dir[0], first_dir[1], 0.0]))
                heading_align = float(np.dot(vel_dir, first_xy))
                # heading_align: 1.0 = aligned, 0 = perpendicular, -1 = opposite
                # Quadratic penalty for misalignment, scaled by speed
                heading_penalty = max(0.0, 0.5 - heading_align) ** 2
                score += 18.0 * speed_xy * heading_penalty

        for i in range(1, len(route) - 1):
            a = self._unit(route[i] - route[i - 1])
            b = self._unit(route[i + 1] - route[i])
            score += 0.45 * max(0.0, 1.0 - float(np.dot(a, b))) ** 2

        sampled = self._sample_polyline(route, spacing=self.ROUTE_SAMPLE_SPACING)
        if self._obstacle_positions.size > 0:
            soft = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER + 0.12
            for op in self._obstacle_positions:
                d = np.linalg.norm(sampled[:, :2] - op[:2], axis=1)
                d_min = float(np.min(d))
                if d_min < soft:
                    score += 180.0 * ((soft - d_min) / soft) ** 2
                if d_min < self.OBSTACLE_RADIUS:
                    score += 1800.0 * ((self.OBSTACLE_RADIUS - d_min) / self.OBSTACLE_RADIUS) ** 2

        z = sampled[:, 2]
        below = np.maximum(self.GROUND_CLEARANCE + 0.04 - z, 0.0)
        above = np.maximum(z - (self.CEILING - 0.06), 0.0)
        score += 300.0 * float(np.mean(below * below + above * above))

        # --- Gate frame proximity penalty ---
        gate_clearance = self._route_min_gate_clearance(route)
        if gate_clearance < self.ROUTE_GATE_SAFE_DIST:
            deficit = self.ROUTE_GATE_SAFE_DIST - gate_clearance
            score += 350.0 * (deficit / self.ROUTE_GATE_SAFE_DIST) ** 2
        if gate_clearance < self.DRONE_RADIUS:
            score += 3000.0  # route physically intersects a gate frame

        # --- Penalise routes that pass through non-target gates ---
        # A route going through another gate's opening risks collision and
        # wastes time.  Check each non-target gate for sign changes along
        # the route (indicating a crossing).
        for gi in range(self._n_gates):
            if gi == self._target_gate:
                continue
            gp_other = self._gate_positions[gi]
            n_other = self._get_gate_normal(gi, sampled[0])
            signed = (sampled - gp_other) @ n_other
            # Check for sign changes (crossing the gate plane)
            for j in range(len(signed) - 1):
                if signed[j] * signed[j + 1] < 0.0:
                    # Crossing detected — check if it's near the gate opening
                    cross_pt = sampled[j]
                    R_other = self._gate_rotmats[gi]
                    local = (cross_pt - gp_other) @ R_other
                    if abs(local[1]) < self.GATE_OUTER_HALF and abs(local[2]) < self.GATE_OUTER_HALF:
                        score += 500.0  # heavy penalty for going through another gate
                    break  # one crossing per gate is enough

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            d0 = float(np.linalg.norm(pos - gp))
            d1 = float(np.linalg.norm(route[min(1, len(route) - 1)] - gp))
            score -= 12.0 * max(0.0, d0 - d1)

        return float(score)

    def _build_route_candidates(self, pos, vel):
        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            route = self._clean_route([pos.copy(), pos.copy()])
            return [route], [0.0]

        gp = self._gate_positions[self._target_gate]
        R = self._gate_rotmats[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)
        gate_y = self._unit(R[:, 1].copy())
        gate_z = self._unit(R[:, 2].copy())

        entry_center = gp - self.FUNNEL_LENGTH * normal
        approach = gp - self.APPROACH_DIST * normal
        exit_pt = gp + self.EXIT_DIST * normal
        tail = self._make_gate_tail(pos)

        candidates = []
        seen: set[tuple[float, ...]] = set()

        def add_route(points):
            route = self._clean_route(points)
            if len(route) < 2:
                return
            key = tuple(np.round(route.reshape(-1), 2).tolist())
            if key in seen:
                return
            seen.add(key)
            candidates.append(route)

        add_route([pos.copy(), gp.copy(), exit_pt] + tail[3:])
        add_route([pos.copy()] + tail)
        add_route([pos.copy(), approach, gp.copy(), exit_pt] + tail[3:])

        for dy in self.ROUTE_ENTRY_Y_OFFSETS:
            for dz in self.ROUTE_ENTRY_Z_OFFSETS:
                if abs(dy) < _EPS and abs(dz) < _EPS:
                    continue
                entry = entry_center + dy * gate_y + dz * gate_z
                app = approach + 0.45 * dy * gate_y + 0.45 * dz * gate_z
                add_route([pos.copy(), entry, app, gp.copy(), exit_pt] + tail[3:])

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

        corridor_target = entry_center if float(np.linalg.norm(entry_center - pos)) > 0.25 else gp
        a_xy, b_xy = pos[:2], corridor_target[:2]
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
                    clearance = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER + 0.18
                    detour_xy = op[:2] + sign * clearance * side_xy
                    detour = np.array([detour_xy[0], detour_xy[1], z_detour], dtype=np.float64)
                    # Smooth approach/exit from detour with intermediate blend points
                    pre_t = max(t_seg - 0.20, 0.05)
                    post_t = min(t_seg + 0.20, 0.95)
                    pre_pt = (1.0 - pre_t) * pos + pre_t * corridor_target
                    post_pt = (1.0 - post_t) * pos + post_t * corridor_target
                    pre_blend = 0.65 * pre_pt + 0.35 * detour
                    post_blend = 0.65 * post_pt + 0.35 * detour
                    add_route([pos.copy(), pre_blend, detour, post_blend, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        speed = float(np.linalg.norm(vel[:2]))
        if speed > 0.25:
            v_dir = self._unit(np.array([vel[0], vel[1], 0.0]), fallback=side)
            look = pos + min(0.55, 0.25 * speed) * v_dir
            look[2] = pos[2]
            add_route([pos.copy(), look, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        # --- Reversal / dip routes ---
        # When the drone is heading AWAY from the gate, build a wide looping
        # route that gives the drone room to brake, turn, and re-approach.
        # The arc gains altitude to provide braking margin.
        if speed > 0.30:
            v_dir = self._unit(np.array([vel[0], vel[1], 0.0]), fallback=side)
            to_gate_dir = self._unit(gp[:2] - pos[:2], fallback=v_dir[:2])
            heading_vs_gate = float(v_dir[0] * to_gate_dir[0] + v_dir[1] * to_gate_dir[1])

            if heading_vs_gate < 0.25:
                # Scale turn radius with speed — faster = wider arc needed
                extend_dist = min(1.00, 0.40 * speed)
                arc_radius = max(0.65, 0.45 * speed)  # wider at higher speed

                fwd_pt = pos.copy()
                fwd_pt[:2] += extend_dist * v_dir[:2]
                # Gain altitude during the forward extension for braking room
                z_gain = min(0.20, 0.12 * speed)
                fwd_pt[2] = min(fwd_pt[2] + z_gain, self.CEILING - 0.10)
                fwd_pt[2] = max(fwd_pt[2], self.GROUND_CLEARANCE + 0.06)

                perp = np.array([-v_dir[1], v_dir[0], 0.0], dtype=np.float64)
                for sign in (-1.0, 1.0):
                    # Arc apex: sideways + even more altitude
                    arc_pt = fwd_pt.copy()
                    arc_pt[:2] += sign * arc_radius * perp[:2]
                    arc_pt[2] = min(fwd_pt[2] + 0.10, self.CEILING - 0.10)

                    # Midpoint between arc apex and entry — smooth the return
                    mid_return = 0.50 * arc_pt + 0.50 * entry_center
                    mid_return[2] = 0.60 * arc_pt[2] + 0.40 * entry_center[2]

                    add_route([pos.copy(), fwd_pt.copy(), arc_pt, mid_return, entry_center, approach, gp.copy(), exit_pt] + tail[3:])

        if not candidates:
            add_route([pos.copy()] + tail)

        # Deflect all candidates away from gate frames, then straighten near gates
        candidates = [self._straighten_near_gates(self._deflect_route_from_gates(r)) for r in candidates]

        scored = [(self._score_route(r, pos, vel), r) for r in candidates]
        scored.sort(key=lambda item: item[0])
        top = scored[:max(1, int(self.ROUTE_PRESELECT_K))]
        return [r for _, r in top], [float(c) for c, _ in top]

    # ------------------------------------------------------------------
    #  Reference generation & deterministic PD sequence (now 4-D)
    # ------------------------------------------------------------------
    def _generate_references(self, pos, vel, route_points=None):
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
        avg_speed = min(self.V_CRUISE, max(0.85 * self.V_CRUISE, 0.5 * (speed0 + self.V_CRUISE)))

        # Pre-compute turn sharpness at each route waypoint for speed profiling
        seg_turn = np.zeros(len(seg_lens), dtype=np.float64)
        for i in range(len(seg_lens)):
            if i > 0 and seg_lens[i] > 0.02 and seg_lens[i - 1] > 0.02:
                d_prev = self._unit(route[i] - route[i - 1])
                d_next = self._unit(route[i + 1] - route[i])
                turn_cos = float(np.dot(d_prev, d_next))
                seg_turn[i] = max(0.0, 1.0 - turn_cos)  # 0=straight, 2=reversal
            if i + 1 < len(seg_lens) and seg_lens[i] > 0.02 and seg_lens[i + 1] > 0.02:
                d_cur = self._unit(route[i + 1] - route[i])
                d_nxt = self._unit(route[i + 2] - route[i + 1])
                turn_cos2 = float(np.dot(d_cur, d_nxt))
                seg_turn[i] = max(seg_turn[i], max(0.0, 1.0 - turn_cos2))

        for k in range(self.MPC_N + 1):
            t = k * self.MPC_DT
            s = min(avg_speed * t, total)
            seg = int(np.clip(int(np.searchsorted(cum, s, side="right") - 1), 0, len(seg_lens) - 1))
            seg_len = max(float(seg_lens[seg]), _EPS)
            alpha = float(np.clip((s - cum[seg]) / seg_len, 0.0, 1.0))
            ref_pos[k] = (1.0 - alpha) * route[seg] + alpha * route[seg + 1]
            ref_pos[k, 2] = float(np.clip(ref_pos[k, 2], self.GROUND_CLEARANCE, self.CEILING))

            if k < self.MPC_N:
                tangent = self._unit(route[seg + 1] - route[seg], fallback=normal)
                dist_to_gate = float(np.linalg.norm(ref_pos[k] - gp))

                # Speed reduction at sharp turns: slow down proportional to turn angle
                turn = float(seg_turn[seg])
                turn_speed_factor = max(0.40, 1.0 - 0.55 * turn)  # 40% speed at full reversal

                if dist_to_gate < self.ALIGN_START_DIST:
                    blend = 1.0 - dist_to_gate / max(self.ALIGN_START_DIST, _EPS)
                    blend = float(np.clip(blend, 0.0, 1.0))
                    tangent = self._unit((1.0 - blend) * tangent + blend * normal, fallback=normal)
                    v_des = self.V_GATE + (self.V_CRUISE - self.V_GATE) * (1.0 - blend)
                else:
                    v_des = self.V_CRUISE
                v_des *= turn_speed_factor
                ref_vel[k] = tangent * v_des
                ref_vel[k, 2] = float(np.clip(ref_vel[k, 2], -self.VZ_REF_MAX, self.VZ_REF_MAX))

        ref_vel[self.MPC_N] = ref_vel[self.MPC_N - 1]
        self._apply_launch_altitude_ramp(ref_pos, ref_vel)
        return ref_pos, ref_vel

    def _make_pd_sequence(self, pos, vel, rpy, drpy, ref_pos, ref_vel):
        """Deterministic PD anchor sequence in 4-D attitude+thrust control space.

        Instead of computing world-frame acceleration and clipping, this computes
        a desired acceleration, then converts it to attitude+thrust commands that
        the full RPY dynamics model can track.
        """
        seq = np.zeros((self.MPC_N, 4), dtype=np.float64)
        p = pos.astype(np.float64).copy()
        v = vel.astype(np.float64).copy()
        r = rpy.astype(np.float64).copy()
        dr = drpy.astype(np.float64).copy()

        kp = np.array([4.7, 4.7, 4.6], dtype=np.float64)
        kd = np.array([2.8, 2.8, 2.5], dtype=np.float64)
        dt = self.MPC_DT

        # Rotor lag state for PD propagation
        T_act_pd = self._thrust_actual if self._has_rotor_lag else None

        for k in range(self.MPC_N):
            e_p = ref_pos[k + 1] - p
            e_v = ref_vel[k + 1] - v

            # Desired world-frame specific thrust
            a_des = np.array([0.0, 0.0, self._g]) + kp * e_p + kd * e_v

            # Convert desired acceleration to attitude commands
            roll_cmd, pitch_cmd, yaw_cmd, thrust = self._accel_to_attitude_cmd(
                a_des, target_yaw=0.0,
            )
            seq[k] = [roll_cmd, pitch_cmd, yaw_cmd, thrust]

            # Propagate state with sub-stepping for numerical stability
            n_sub = self.ROLLOUT_SUB_STEPS
            sub_dt = dt / n_sub
            cmd_arr = np.array([roll_cmd, pitch_cmd, yaw_cmd])
            inv_m = 1.0 / self._mass_estimate

            for _ in range(n_sub):
                # Rotor lag
                if self._has_rotor_lag:
                    T_act_pd += (thrust - T_act_pd) * (sub_dt / self._thrust_time_coef)
                    thrust_phys = self._acc_coef + self._cmd_f_coef * T_act_pd
                else:
                    thrust_phys = self._acc_coef + self._cmd_f_coef * thrust

                cx, sx = math.cos(r[0]), math.sin(r[0])
                cy, sy = math.cos(r[1]), math.sin(r[1])
                cz, sz = math.cos(r[2]), math.sin(r[2])
                R_02 = sx * sz + cx * cz * sy
                R_12 = cx * sy * sz - cz * sx
                R_22 = cx * cy
                acc_w = np.array([
                    R_02 * thrust_phys * inv_m,
                    R_12 * thrust_phys * inv_m,
                    -self._g + R_22 * thrust_phys * inv_m,
                ])

                # Drag
                if self._has_drag:
                    R_00 = cz * cy
                    R_01 = cz * sy * sx - sz * cx
                    R_10 = sz * cy
                    R_11 = sz * sy * sx + cz * cx
                    R_20 = -sy
                    R_21 = cy * sx
                    vb_x = R_00 * v[0] + R_10 * v[1] + R_20 * v[2]
                    vb_y = R_01 * v[0] + R_11 * v[1] + R_21 * v[2]
                    vb_z = R_02 * v[0] + R_12 * v[1] + R_22 * v[2]
                    fd_x = self._drag_xy * vb_x
                    fd_y = self._drag_xy * vb_y
                    fd_z = self._drag_z * vb_z
                    acc_w[0] += (R_00 * fd_x + R_01 * fd_y + R_02 * fd_z) * inv_m
                    acc_w[1] += (R_10 * fd_x + R_11 * fd_y + R_12 * fd_z) * inv_m
                    acc_w[2] += (R_20 * fd_x + R_21 * fd_y + R_22 * fd_z) * inv_m

                v = v + acc_w * sub_dt
                p = p + v * sub_dt
                ddr = self._rpy_coef * r + self._rpy_rates_coef * dr + self._cmd_rpy_coef * cmd_arr
                dr = dr + ddr * sub_dt
                r = r + dr * sub_dt

        return self._clip_u(seq)

    def _accel_to_attitude_cmd(self, a_des, target_yaw=0.0):
        """Convert desired world-frame specific-thrust vector to attitude+thrust.

        This is used only for the deterministic PD warm-start, not for the final
        output. The MPPI rollout samples directly in attitude space.
        """
        a = np.asarray(a_des, dtype=np.float64)
        a[2] = max(a[2], 0.1)
        a_mag = float(np.linalg.norm(a))
        if a_mag < 0.5:
            return 0.0, 0.0, target_yaw, self._hover_thrust

        z_des = a / a_mag
        cy, sy = math.cos(target_yaw), math.sin(target_yaw)
        zx = cy * z_des[0] + sy * z_des[1]
        zy = -sy * z_des[0] + cy * z_des[1]
        zz = z_des[2]

        roll = float(np.arctan2(-zy, zz))
        pitch = float(np.arctan2(zx, math.sqrt(zy * zy + zz * zz)))
        roll = float(np.clip(roll, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        pitch = float(np.clip(pitch, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))

        # Vertical-preserving thrust (command value, not physical)
        c = max(math.cos(roll) * math.cos(pitch), 0.25)
        thrust = (self._mass_estimate * max(a[2], 0.1) / c - self._acc_coef) / self._cmd_f_coef
        thrust = float(np.clip(thrust, self._thrust_min, self._thrust_max))

        return roll, pitch, target_yaw, thrust

    # ------------------------------------------------------------------
    #  Scoring with full dynamics + spatial costs
    # ------------------------------------------------------------------
    def _score_rollouts(
        self,
        samples: np.ndarray,
        pos0: np.ndarray,
        vel0: np.ndarray,
        rpy0: np.ndarray,
        drpy0: np.ndarray,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Roll out all samples with full RPY dynamics and score with spatial costs."""
        K, N, _ = samples.shape

        # --- Full RPY+thrust forward integration ---
        all_pos, all_vel, all_rpy, all_drpy = self._rollout_rpy_thrust(
            samples, pos0, vel0, rpy0, drpy0,
        )

        cost = np.zeros(K, dtype=np.float64)
        min_margin = np.full(K, np.inf, dtype=np.float64)

        target_valid = 0 <= self._target_gate < self._n_gates
        if target_valid:
            gp = self._gate_positions[self._target_gate]
            R = self._gate_rotmats[self._target_gate]
            normal = self._get_gate_normal(self._target_gate, pos0)
            signed0 = float((pos0 - gp) @ normal)
            signed_prev = np.full(K, signed0, dtype=np.float64)
            max_progress = signed_prev.copy()
            good_crossed = np.zeros(K, dtype=bool)
            bad_crossed = np.zeros(K, dtype=bool)
            dist0_gate = float(np.linalg.norm(pos0 - gp))
            reachable = dist0_gate < max(0.85, 1.10 * self.V_MAX * N * self.MPC_DT)
        else:
            gp = np.zeros(3)
            R = np.eye(3)
            normal = np.array([1.0, 0.0, 0.0])
            signed_prev = np.zeros(K)
            max_progress = np.zeros(K)
            good_crossed = np.zeros(K, dtype=bool)
            bad_crossed = np.zeros(K, dtype=bool)
            reachable = False
            signed0 = 0.0
            dist0_gate = 0.0

        next_valid = target_valid and (self._target_gate + 1 < self._n_gates)
        gp_next = self._gate_positions[self._target_gate + 1] if next_valid else None
        R_next = self._gate_rotmats[self._target_gate + 1] if next_valid else None

        prev_valid = 0 <= self._prev_target_gate < self._n_gates
        gp_prev = self._gate_positions[self._prev_target_gate] if prev_valid else None
        R_prev = self._gate_rotmats[self._prev_target_gate] if prev_valid else None

        last_u = np.tile(self._prev_u[:4], (K, 1))

        for k in range(N):
            u = samples[:, k, :]
            du = u - last_u
            last_u = u

            pos = all_pos[:, k + 1]
            vel = all_vel[:, k + 1]
            rpy_k = all_rpy[:, k + 1]
            pos_prev = all_pos[:, k]

            stage = (k + 1) / N
            stage_w = 0.65 + 0.70 * stage

            # --- Control regularisation ---
            cost += self.W_INPUT * np.sum((u - self._hover_u) ** 2, axis=1)
            cost += self.W_DINPUT * np.sum(du ** 2, axis=1)

            # --- Attitude penalty (NEW: penalise large roll/pitch) ---
            roll_abs = np.abs(rpy_k[:, 0])
            pitch_abs = np.abs(rpy_k[:, 1])
            cost += self.W_ATTITUDE_LIMIT * np.maximum(roll_abs - self.MAX_ROLL_PITCH_CMD, 0.0) ** 2
            cost += self.W_ATTITUDE_LIMIT * np.maximum(pitch_abs - self.MAX_ROLL_PITCH_CMD, 0.0) ** 2
            cost += self.W_ATTITUDE_SMOOTH * np.sum(all_drpy[:, k + 1] ** 2, axis=1)

            # --- Reference tracking ---
            e_p = pos - ref_pos[k + 1]
            e_v = vel - ref_vel[k + 1]
            cost += stage_w * (
                self.W_REF_POS * np.sum(e_p ** 2, axis=1)
                + self.W_REF_VEL * np.sum(e_v ** 2, axis=1)
            )

            # --- Altitude safety ---
            below = np.maximum(self.GROUND_CLEARANCE - pos[:, 2], 0.0)
            above = np.maximum(pos[:, 2] - self.CEILING, 0.0)
            cost += self.W_ALTITUDE * (below ** 2 + above ** 2)
            hard_alt = (pos[:, 2] < 0.02) | (pos[:, 2] > self.CEILING + 0.25)
            cost += self.W_ALTITUDE_HARD * hard_alt
            min_margin = np.minimum(min_margin, pos[:, 2] - self.GROUND_CLEARANCE)
            min_margin = np.minimum(min_margin, self.CEILING - pos[:, 2])

            # --- Speed limit ---
            speed = np.linalg.norm(vel, axis=1)
            overspeed = np.maximum(speed - self.V_MAX, 0.0)
            cost += self.W_SPEED_LIMIT * overspeed ** 2

            # --- Spatial corridor cost (NEW) ---
            if self._geo is not None:
                corridor_cost = self._spatial_corridor_cost_batch(pos)
                cost += corridor_cost

            # --- Spatial progress reward (NEW) ---
            if self._geo is not None and k == N - 1:
                progress, curv_cost = self._spatial_progress_and_curvature(pos, vel)
                cost -= self.W_SPATIAL_PROGRESS * progress
                cost += self.W_CURVATURE_SPEED * curv_cost

            # --- Cylindrical obstacle avoidance (batched) ---
            if self._n_obstacles > 0:
                # (K, n_obs, 2) distance vectors
                diff_xy = pos[:, None, :2] - self._obstacle_positions[None, :, :2]
                d = np.sqrt(np.maximum(np.sum(diff_xy ** 2, axis=2), _EPS))  # (K, n_obs)
                soft_r = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER
                v_soft = np.maximum(soft_r - d, 0.0)
                v_hard = np.maximum(self.OBSTACLE_RADIUS - d, 0.0)
                cost += self.W_POLE_BUFFER * np.sum((v_soft / soft_r) ** 2, axis=1)
                cost += self.W_POLE_COLLISION * np.sum((v_hard / self.OBSTACLE_RADIUS) ** 2, axis=1)
                cost += self.W_POLE_NEAR_EXP * np.sum(np.exp(-np.maximum(d - self.OBSTACLE_RADIUS, 0.0) / 0.18), axis=1)
                min_margin = np.minimum(min_margin, np.min(d - self.OBSTACLE_RADIUS, axis=1))

            # --- Gate geometry ---
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
                    c_gate_p, m_gate_p = self._gate_geometry_cost_and_margin(
                        pos[~still_approaching], gp, R, active_funnel=False,
                    )
                    cost[~still_approaching] += 0.85 * c_gate_p
                    min_margin[~still_approaching] = np.minimum(min_margin[~still_approaching], m_gate_p)

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
                    y_c = local_cross[:, 1]
                    z_c = local_cross[:, 2]
                    err = y_c * y_c + z_c * z_c
                    clear = max(self.GATE_HALF_OPENING - self.GATE_CLEARANCE - self.DRONE_RADIUS, 0.02)
                    inside = (np.abs(y_c) <= clear) & (np.abs(z_c) <= clear)

                    cost += crossing * self.W_CROSS_CENTER * err
                    cost += crossing * inside * (-self.BONUS_GOOD_CROSS)
                    bad_amount = (np.maximum(np.abs(y_c) - clear, 0.0) ** 2
                                  + np.maximum(np.abs(z_c) - clear, 0.0) ** 2)
                    cost += crossing * (~inside) * (self.W_BAD_CROSS * (1.0 + bad_amount / (clear * clear)))
                    good_crossed |= crossing & inside
                    bad_crossed |= crossing & (~inside)

                signed_prev = signed_now

            if next_valid and gp_next is not None and R_next is not None:
                use_next = good_crossed | (signed_now > self.EXIT_DIST) if target_valid else np.zeros(K, dtype=bool)
                if np.any(use_next):
                    cn, mn = self._gate_geometry_cost_and_margin(pos[use_next], gp_next, R_next, True)
                    cost[use_next] += 0.85 * cn
                    min_margin[use_next] = np.minimum(min_margin[use_next], mn)
                if np.any(~use_next):
                    cn2, mn2 = self._gate_geometry_cost_and_margin(pos[~use_next], gp_next, R_next, False)
                    cost[~use_next] += 0.70 * cn2
                    min_margin[~use_next] = np.minimum(min_margin[~use_next], mn2)

            # --- Previous (just-passed) gate frame ---
            # The drone is physically right at the previous gate's exit and
            # turning toward the next one. Keeping the previous frame in the
            # cost stops it from grazing the exit-side bar. Weight is kept
            # low (0.20): at 0.50 it dominated the trajectory cost near the
            # previous gate, biasing the planner away from useful paths and
            # slowing finishes. The cost decays naturally to zero as the
            # drone moves away.
            if prev_valid and gp_prev is not None and R_prev is not None:
                cp, mp = self._gate_geometry_cost_and_margin(
                    pos, gp_prev, R_prev, active_funnel=False,
                )
                cost += 0.20 * cp
                min_margin = np.minimum(min_margin, mp)

        # --- Terminal ---
        terminal_pos = all_pos[:, -1]
        terminal_error = terminal_pos - ref_pos[-1]
        cost += self.W_TERMINAL_REF * np.sum(terminal_error ** 2, axis=1)

        if target_valid:
            signed_final = (terminal_pos - gp) @ normal
            cost -= self.W_PROGRESS * (max_progress - signed0)
            dist_final = np.linalg.norm(terminal_pos - gp, axis=1)
            miss = ~good_crossed
            cost += miss * self.W_GATE_DISTANCE_TERMINAL * (dist_final / max(dist0_gate, 1.0)) ** 2
            cost -= good_crossed * 250.0

            if reachable:
                cost += self.W_NOT_CROSSED_REACHABLE * miss * np.maximum(0.15 - signed_final, 0.0) ** 2

        min_margin = np.where(bad_crossed, np.minimum(min_margin, -0.10), min_margin)

        cost = np.nan_to_num(cost, nan=1e12, posinf=1e12, neginf=-1e12)
        min_margin = np.nan_to_num(min_margin, nan=-1.0, posinf=1e6, neginf=-1.0)

        return cost, min_margin, all_pos.astype(np.float32), good_crossed, bad_crossed

    # ------------------------------------------------------------------
    #  Sampling in 4-D control space
    # ------------------------------------------------------------------
    def _sample_sequences(self, route_seqs):
        K = int(self.K_SAMPLES)
        N = int(self.MPC_N)

        cleaned = []
        for seq in route_seqs:
            arr = self._clip_u(np.asarray(seq, dtype=np.float64))
            if arr.shape == (N, 4):
                cleaned.append(arr)
        if not cleaned:
            cleaned = [np.tile(self._hover_u, (N, 1))]

        if self.FORGET_OLD_SCENARIOS:
            sigma = self._base_sigma.copy()
            self._sigma = sigma.copy()
        else:
            sigma = self._sigma

        n_det = min(max(5, 2 * len(cleaned) + 3), max(1, K // 2))
        n_rand = max(K - n_det, 1)
        half = max(n_rand // 2, 1)

        eps_half = self._rng.standard_normal((half, N, 4)) * sigma[None, :, :]
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        while eps.shape[0] < n_rand:
            extra = self._rng.standard_normal((1, N, 4)) * sigma[None, :, :]
            eps = np.concatenate([eps, extra], axis=0)
        eps = eps[:n_rand]

        rho = float(self.NOISE_RHO)
        c = math.sqrt(max(1.0 - rho * rho, 0.0))
        smooth = np.empty_like(eps)
        smooth[:, 0, :] = eps[:, 0, :]
        for k in range(1, N):
            smooth[:, k, :] = rho * smooth[:, k - 1, :] + c * eps[:, k, :]

        center_bank = cleaned[:max(1, min(len(cleaned), self.ROUTE_TOP_K))]
        center_arr = np.stack(center_bank, axis=0)
        probs = np.exp(-0.70 * np.arange(len(center_bank), dtype=np.float64))
        probs /= max(float(np.sum(probs)), _EPS)
        center_idx = self._rng.choice(len(center_bank), size=n_rand, p=probs)

        samples = np.empty((K, N, 4), dtype=np.float64)
        samples[:n_rand] = center_arr[center_idx] + smooth

        deterministic = [cleaned[0].copy()]
        for seq in cleaned[1:]:
            deterministic.append(seq.copy())
            deterministic.append(0.70 * cleaned[0] + 0.30 * seq)

        # Vertical variants
        up = cleaned[0].copy()
        down = cleaned[0].copy()
        up[:, 3] += 0.004   # slightly more thrust
        down[:, 3] -= 0.004  # slightly less thrust
        deterministic.extend([up, down])

        write = n_rand
        det_i = 0
        while write < K:
            samples[write] = deterministic[det_i % len(deterministic)]
            write += 1
            det_i += 1

        return self._clip_u(samples)

    def _shift_distribution(self):
        self._mean_u[:-1] = self._mean_u[1:]
        self._mean_u[-1] = self._mean_u[-2]
        self._mean_u = self._clip_u(self._mean_u)
        self._sigma[:-1] = self._sigma[1:]
        self._sigma[-1] = np.maximum(self._sigma[-2], self.SIGMA_INIT)
        self._sigma = np.clip(self._sigma, self.SIGMA_MIN, self.SIGMA_MAX)

    def _update_distribution(self, samples, cost, min_margin):
        safe_mask = min_margin > 0.0
        fit_idx = np.where(safe_mask)[0] if np.any(safe_mask) else np.arange(samples.shape[0])

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

        elite_count = min(int(self.N_ELITES), len(fit_idx))
        elite_local = np.argsort(c)[:elite_count]
        elite_idx = fit_idx[elite_local]
        elite_delta = samples[elite_idx] - new_mean[None, :, :]
        elite_std = np.sqrt(np.mean(elite_delta ** 2, axis=0) + 1e-8)
        target_sigma = np.clip(1.35 * elite_std, self.SIGMA_MIN, self.SIGMA_MAX)
        self._sigma = np.clip(0.82 * self._sigma + 0.18 * target_sigma, self.SIGMA_MIN, self.SIGMA_MAX)
        self._mean_u = new_mean

        return new_mean, best_idx, bool(np.any(safe_mask))

    # ------------------------------------------------------------------
    #  Candidate selection (from scenario MPC)
    # ------------------------------------------------------------------
    def _crossing_yz_from_trajectories(self, trajectories, pos0):
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
            s0, s1 = signed[:, k], signed[:, k + 1]
            mask = (~crossed) & (s0 <= 0.0) & (s1 >= 0.0)
            if not np.any(mask):
                continue
            denom = np.maximum(s1[mask] - s0[mask], _EPS)
            alpha = np.clip(-s0[mask] / denom, 0.0, 1.0)
            p_cross = pts[mask, k, :] + alpha[:, None] * (pts[mask, k + 1, :] - pts[mask, k, :])
            local = self._world_to_gate_local(p_cross, gp, R)
            yz_radius[mask] = np.sqrt(local[:, 1] ** 2 + local[:, 2] ** 2)
            crossed[mask] = True
        return yz_radius, crossed

    def _candidate_progress_cost(self, indices, cost, trajectories, pos, min_margin, cross_yz=None):
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        select_cost = cost[indices].astype(np.float64).copy()
        select_cost += 9000.0 * np.maximum(self.SELECTION_MARGIN - min_margin[indices], 0.0) ** 2
        select_cost -= 140.0 * np.maximum(min_margin[indices] - self.SELECTION_MARGIN, 0.0)
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

    def _choose_fresh_robust_candidates(self, cost, min_margin, trajectories, good_crossed, bad_crossed, pos):
        cross_yz, geometric_crossed = self._crossing_yz_from_trajectories(trajectories, pos)
        noncrash = (~bad_crossed) & (min_margin > self.SAFE_SELECTION_MARGIN)
        robust = (~bad_crossed) & (min_margin > self.SELECTION_MARGIN)

        for label, mask in [
            ("robust_center_cross", robust & good_crossed & geometric_crossed & (cross_yz <= self.ROBUST_CROSS_RADIUS)),
            ("robust_cross", robust & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS)),
            ("loose_center_cross", noncrash & good_crossed & geometric_crossed & (cross_yz <= self.ROBUST_CROSS_RADIUS)),
            ("loose_cross", noncrash & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS)),
            ("robust_progress", robust),
            ("loose_progress", noncrash),
        ]:
            if np.any(mask):
                pool = np.where(mask)[0]
                sel_cost = self._candidate_progress_cost(pool, cost, trajectories, pos, min_margin, cross_yz)
                order = pool[np.argsort(sel_cost)]
                keep = max(1, min(int(self.ELITE_BLEND_COUNT), len(order)))
                return order[:keep].astype(int), label

        pool = np.arange(cost.shape[0])
        sel_cost = self._candidate_progress_cost(pool, cost, trajectories, pos, min_margin, cross_yz)
        order = pool[np.lexsort((sel_cost, -min_margin[pool]))]
        return order[:1].astype(int), "emergency"

    def _elite_weighted_sequence(self, samples, cost, candidate_idx):
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

    # ------------------------------------------------------------------
    #  Route-ranking by rollout (4-D)
    # ------------------------------------------------------------------
    def _rank_routes_by_rollout(self, pos, vel, rpy, drpy, routes, route_costs):
        if not routes:
            ref_pos, ref_vel = self._generate_references(pos, vel, None)
            seq = self._make_pd_sequence(pos, vel, rpy, drpy, ref_pos, ref_vel)
            return [], [], [seq], ref_pos, ref_vel

        records = []
        for idx, route in enumerate(routes):
            r_pos, r_vel = self._generate_references(pos, vel, route)
            seq = self._make_pd_sequence(pos, vel, rpy, drpy, r_pos, r_vel)
            c, margin, traj, crossed, bad = self._score_rollouts(
                seq[None, :, :], pos, vel, rpy, drpy, r_pos, r_vel,
            )
            c0, m0 = float(c[0]), float(margin[0])
            crossed0, bad0 = bool(crossed[0]), bool(bad[0])
            geom = float(route_costs[idx]) if idx < len(route_costs) else 0.0

            select_score = c0 + 0.04 * geom
            if crossed0 and not bad0 and m0 > -0.01:
                select_score -= 4500.0
            if not crossed0:
                select_score += 850.0
            if bad0:
                select_score += 20000.0
            if m0 < 0.0:
                select_score += 12000.0 * (0.02 - m0) ** 2 + 2500.0

            bucket = 0 if (crossed0 and not bad0 and m0 > 0.0) else (1 if m0 > 0.0 else 2)
            records.append((
                bucket, select_score, idx, route, geom, seq, r_pos, r_vel,
                c0, m0, crossed0, bad0, traj[0].astype(np.float64),
            ))

        records.sort(key=lambda item: (item[0], item[1]))
        keep = records[:max(1, int(self.ROUTE_TOP_K))]
        routes_ranked = [r[3] for r in keep]
        costs_ranked = [float(r[4]) for r in keep]
        seqs_ranked = [r[5] for r in keep]
        best = keep[0]
        self._last_route_cost = float(best[4])
        self._last_route_anchor_cost = float(best[8])
        self._last_route_anchor_margin = float(best[9])
        self._last_route_anchor_crossed = bool(best[10])
        self._last_route_anchor_traj = best[12]
        return routes_ranked, costs_ranked, seqs_ranked, best[6], best[7]

    # ------------------------------------------------------------------
    #  Output helpers
    # ------------------------------------------------------------------
    def _reactive_push_attitude(self, pos: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """Small obstacle APF push translated into attitude adjustments."""
        push_accel = np.zeros(3, dtype=np.float64)
        for op in self._obstacle_positions:
            diff = pos[:2] - op[:2]
            d = float(np.linalg.norm(diff))
            if _EPS < d < self.APF_INFLUENCE:
                mag = self.APF_GAIN * (1.0 / d - 1.0 / self.APF_INFLUENCE) / (d * d)
                mag = min(float(mag), self.APF_MAX)
                push_accel[:2] += mag * diff / d

        # Convert push acceleration to attitude perturbation
        if float(np.linalg.norm(push_accel[:2])) > _EPS:
            # Small-angle: desired roll ~ -ay/g, desired pitch ~ ax/g
            d_roll = float(np.clip(-push_accel[1] / self._g, -0.08, 0.08))
            d_pitch = float(np.clip(push_accel[0] / self._g, -0.08, 0.08))
            out = u0.copy()
            out[0] = float(np.clip(out[0] + d_roll, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
            out[1] = float(np.clip(out[1] + d_pitch, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
            return out
        return u0.copy()

    def _yaw_command(self, obs, pos):
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

    def _finish_action(self, obs, pos, u0):
        """Convert the 4-D MPPI command directly to simulator action.

        Unlike the original scenario MPC which uses a lossy accel-to-attitude
        conversion, this outputs the planned attitude+thrust directly.
        No yaw override — in attitude-space planning, the yaw is part of the
        rotation matrix and must match what the rollout predicted.
        """
        roll_cmd = float(np.clip(u0[0], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        pitch_cmd = float(np.clip(u0[1], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        yaw_cmd = float(np.clip(u0[2], -self.MAX_YAW_CMD, self.MAX_YAW_CMD))
        thrust_cmd = float(np.clip(u0[3], self._thrust_min, self._thrust_max))

        # Altitude safety guards
        if pos[2] < self.GROUND_CLEARANCE + 0.08:
            thrust_cmd += 0.65 * (self.GROUND_CLEARANCE + 0.08 - pos[2]) * self._mass_estimate * self._g
        if pos[2] > self.CEILING - 0.18:
            overshoot = pos[2] - (self.CEILING - 0.18)
            thrust_cmd -= (0.55 + 2.2 * overshoot) * overshoot * self._mass_estimate * self._g
        thrust_cmd = float(np.clip(thrust_cmd, self._thrust_min, self._thrust_max))

        self._prev_output = np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd], dtype=np.float64)
        return self._prev_output.astype(np.float32)

    # ------------------------------------------------------------------
    #  Route tracking feedback (translated to attitude space)
    # ------------------------------------------------------------------
    def _route_tracking_feedback(self, u_base, pos, vel, rpy, elapsed, ref_pos=None, ref_vel=None):
        """Closed-loop correction in attitude+thrust space toward the selected route."""
        if ref_pos is None:
            ref_pos = self._cached_ref_pos
        if ref_vel is None:
            ref_vel = self._cached_ref_vel

        # Compute desired position error and velocity error
        path = None
        if self.USE_PROJECTED_ROUTE_TRACKING:
            if self._last_route_anchor_traj is not None and len(self._last_route_anchor_traj) >= 3:
                path = np.asarray(self._last_route_anchor_traj, dtype=np.float64)
            elif ref_pos is not None and len(ref_pos) >= 3:
                path = np.asarray(ref_pos, dtype=np.float64)

        if path is None or len(path) < 2:
            # Simple time-indexed reference tracking
            tau = max(0.0, float(elapsed) + self.TRACK_LOOKAHEAD)
            s = tau / max(self.MPC_DT, _EPS)
            k0 = int(np.clip(math.floor(s), 0, self.MPC_N - 1))
            k1 = min(k0 + 1, self.MPC_N)
            a = float(np.clip(s - k0, 0.0, 1.0))
            p_ref = (1.0 - a) * ref_pos[k0] + a * ref_pos[k1]
            v_ref = (1.0 - a) * ref_vel[k0] + a * ref_vel[k1]
            e_p = p_ref - pos
            e_v = v_ref - vel
        else:
            speed = float(np.linalg.norm(vel))
            lookahead = self.PATH_LOOKAHEAD_DIST + self.PATH_LOOKAHEAD_SPEED_GAIN * speed
            lookahead = float(np.clip(lookahead, self.PATH_LOOKAHEAD_DIST, self.PATH_LOOKAHEAD_MAX))
            p_ref, tangent, seg_idx, _ = self._project_polyline_tracking_target(path, pos, lookahead)
            e_p = p_ref - pos
            e_v = np.zeros(3, dtype=np.float64)
            # Desired along-track velocity
            desired_speed = self.V_CRUISE
            if 0 <= self._target_gate < self._n_gates:
                d_gate = float(np.linalg.norm(pos - self._gate_positions[self._target_gate]))
                if d_gate < self.ALIGN_START_DIST:
                    blend = 1.0 - d_gate / max(self.ALIGN_START_DIST, _EPS)
                    desired_speed = min(desired_speed, self.V_CRUISE - blend * (self.V_CRUISE - self.V_GATE))
            e_v = tangent * desired_speed - vel

        # Convert position/velocity error to desired acceleration correction
        kp_xy, kp_z = 3.2, 1.8
        kd_xy, kd_z = 2.0, 1.2
        a_corr = np.array([
            kp_xy * e_p[0] + kd_xy * e_v[0],
            kp_xy * e_p[1] + kd_xy * e_v[1],
            kp_z * e_p[2] + kd_z * e_v[2],
        ])

        # Convert acceleration correction to attitude perturbation (small-angle)
        # roll ~ -ay / g, pitch ~ ax / g
        d_roll = float(np.clip(-a_corr[1] / self._g, -0.15, 0.15))
        d_pitch = float(np.clip(a_corr[0] / self._g, -0.15, 0.15))
        d_thrust = float(np.clip(a_corr[2] * self._mass_estimate, -0.03, 0.03))

        out = np.asarray(u_base, dtype=np.float64).copy()
        out[0] = float(np.clip(out[0] + d_roll, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        out[1] = float(np.clip(out[1] + d_pitch, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        out[3] = float(np.clip(out[3] + d_thrust, self._thrust_min, self._thrust_max))
        return out

    def _project_polyline_tracking_target(self, polyline, pos, lookahead_dist):
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return pos.copy(), np.array([1.0, 0.0, 0.0]), 0, 0.0

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        best_d2, best_i, best_t, best_s = math.inf, 0, 0.0, 0.0
        for i in range(len(seg)):
            if seg_len[i] <= 1e-6:
                continue
            v = seg[i]
            t = float(np.clip(np.dot(pos - pts[i], v) / max(float(np.dot(v, v)), _EPS), 0.0, 1.0))
            q = pts[i] + t * v
            d2 = float(np.dot(pos - q, pos - q))
            if d2 < best_d2:
                best_d2, best_i, best_t = d2, i, t
                best_s = float(cum[i] + t * seg_len[i])

        target_s = min(best_s + max(float(lookahead_dist), 0.02), float(cum[-1]))
        j = int(np.clip(int(np.searchsorted(cum, target_s, side="right") - 1), 0, len(seg_len) - 1))
        denom = max(float(seg_len[j]), _EPS)
        a = float(np.clip((target_s - cum[j]) / denom, 0.0, 1.0))
        target = (1.0 - a) * pts[j] + a * pts[j + 1]
        tangent = self._unit(seg[j], fallback=seg[best_i])
        return target.astype(np.float64), tangent.astype(np.float64), j, best_s

    # ------------------------------------------------------------------
    #  Main controller API
    # ------------------------------------------------------------------
    def compute_control(
        self,
        obs: dict[str, "NDArray[np.floating]"],
        info: dict | None = None,
    ) -> "NDArray[np.floating]":
        self._tick += 1
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        self._last_pos = pos.copy()

        # Extract RPY and angular rates
        quat = np.asarray(obs["quat"], dtype=np.float64)
        rpy = Rot.from_quat(quat).as_euler("xyz").astype(np.float64)
        ang_vel = np.asarray(obs["ang_vel"], dtype=np.float64)
        # Convert body angular velocity to RPY rates (small-angle approx)
        drpy = ang_vel.copy()  # good approximation for small angles
        self._last_rpy = rpy.copy()
        self._last_drpy = drpy.copy()

        # --- Thrust-lag state tracking (exact first-order ODE solution) ---
        if self._has_rotor_lag:
            decay = math.exp(-self._dt / self._thrust_time_coef)
            self._thrust_actual = decay * self._thrust_actual + (1.0 - decay) * self._prev_u[3]

        # --- Online mass estimation (EMA from near-level flight) ---
        if self._tick > 5 and abs(rpy[0]) < 0.30 and abs(rpy[1]) < 0.30:
            T_phys_prev = self._acc_coef + self._cmd_f_coef * self._prev_u[3]
            R22_est = math.cos(rpy[0]) * math.cos(rpy[1])
            if R22_est > 0.5 and T_phys_prev > 0.05:
                m_est = T_phys_prev * R22_est / self._g
                if 0.020 < m_est < 0.100:
                    self._mass_estimate = 0.98 * self._mass_estimate + 0.02 * m_est
                    self._hover_thrust = self._mass_estimate * self._g / self._cmd_f_coef
                    self._hover_u[3] = self._hover_thrust

        if self._launch_z is None:
            self._launch_z = float(pos[2])

        if self._finished:
            return np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float32)

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
                return np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float32)

        do_replan = (self._tick - self._last_planner_tick) >= self._planner_interval_steps

        # --- Fast path: reuse cached 4-D command ---
        if not do_replan:
            elapsed = max(0.0, (self._tick - self._cached_plan_tick) * self._dt)
            idx = min(int(elapsed / max(self.MPC_DT, _EPS)), self.MPC_N - 1)
            u0 = self._cached_u_sequence[idx].copy()
            self._cached_u_index = idx

            u0 = self._route_tracking_feedback(u0, pos, vel, rpy, elapsed)
            u0 = self.CMD_FILTER_ALPHA * u0 + (1.0 - self.CMD_FILTER_ALPHA) * self._prev_u
            u0 = self._reactive_push_attitude(pos, u0)
            u0 = self._clip_u(u0)

            self._prev_u = u0.copy()
            return self._finish_action(obs, pos, u0)

        # --- Expensive replan ---
        plan_t0 = time.perf_counter()
        previous_plan_tick = self._cached_plan_tick
        self._last_planner_tick = self._tick

        if not self.FORGET_OLD_SCENARIOS and previous_plan_tick > -10**8:
            elapsed = max(0.0, (self._tick - previous_plan_tick) * self._dt)
            shift_steps = min(int(elapsed / max(self.MPC_DT, _EPS)), self.MPC_N - 1)
            for _ in range(shift_steps):
                self._shift_distribution()

        # Build and rank route candidates
        raw_routes, raw_route_costs = self._build_route_candidates(pos, vel)
        routes, route_costs, route_seqs, ref_pos, ref_vel = self._rank_routes_by_rollout(
            pos, vel, rpy, drpy, raw_routes, raw_route_costs,
        )

        self._route_candidates = routes
        self._route_costs = route_costs
        self._active_route_points = routes[0].copy() if routes else None

        if route_seqs:
            if self.FORGET_OLD_SCENARIOS:
                self._mean_u = self._clip_u(route_seqs[0].copy())
            else:
                self._mean_u = self._clip_u(
                    (1.0 - self.ROUTE_MEAN_BLEND) * self._mean_u + self.ROUTE_MEAN_BLEND * route_seqs[0],
                )

        # Inject the warped (discovery-shifted) cached sequence as a candidate
        # so MPPI can choose between the fresh plan and the smoothly adapted old plan.
        if hasattr(self, "_warped_discovery_seq") and self._warped_discovery_seq is not None:
            route_seqs.append(self._warped_discovery_seq)
            self._warped_discovery_seq = None

        # Sample and score with full RPY dynamics
        samples = self._sample_sequences(route_seqs)
        cost, min_margin, trajectories, good_crossed, bad_crossed = self._score_rollouts(
            samples, pos, vel, rpy, drpy, ref_pos, ref_vel,
        )
        new_mean, best_idx, any_safe = self._update_distribution(samples, cost, min_margin)

        candidate_idx, selection_label = self._choose_fresh_robust_candidates(
            cost, min_margin, trajectories, good_crossed, bad_crossed, pos,
        )
        selected_sequence, selected_idx, elite_count = self._elite_weighted_sequence(
            samples, cost, candidate_idx,
        )

        chosen_u0 = selected_sequence[0].copy()
        route_u0 = route_seqs[0][0].copy() if route_seqs else chosen_u0.copy()
        best_margin = float(min_margin[selected_idx])
        robust_selected = best_margin > self.SELECTION_MARGIN and not bool(bad_crossed[selected_idx])

        if robust_selected:
            u0 = 0.90 * chosen_u0 + 0.10 * route_u0
        elif min_margin[selected_idx] > 0.0 and not bool(bad_crossed[selected_idx]):
            u0 = 0.82 * chosen_u0 + 0.18 * route_u0
        else:
            u0 = route_u0.copy()

        if route_seqs:
            u0 = (1.0 - self.ROUTE_COMMAND_BLEND) * u0 + self.ROUTE_COMMAND_BLEND * route_u0

        u0 = self._route_tracking_feedback(u0, pos, vel, rpy, 0.0, ref_pos, ref_vel)

        self._mean_u = self._clip_u(
            selected_sequence.copy() if self.FORGET_OLD_SCENARIOS
            else 0.70 * selected_sequence + 0.30 * new_mean,
        )

        self._cached_u_sequence = selected_sequence.copy()
        self._cached_ref_pos = ref_pos.copy()
        self._cached_ref_vel = ref_vel.copy()
        self._cached_plan_tick = self._tick
        self._cached_u_index = 0
        self._cached_u0 = selected_sequence[0].copy()
        self._last_elite_count = int(elite_count)

        u0 = self.CMD_FILTER_ALPHA * u0 + (1.0 - self.CMD_FILTER_ALPHA) * self._prev_u
        u0 = self._reactive_push_attitude(pos, u0)
        u0 = self._clip_u(u0)

        self._prev_u = u0.copy()
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

        if (
            self.DEBUG_PRINT_ENABLED
            and self._plan_counter % max(1, int(self.DEBUG_EVERY_PLANS)) == 0
        ):
            speed = float(np.linalg.norm(vel))
            signed_d = ""
            if 0 <= self._target_gate < self._n_gates:
                gp_ = self._gate_positions[self._target_gate]
                n_ = self._get_gate_normal(self._target_gate, pos)
                sd_ = float((pos - gp_) @ n_)
                dg_ = float(np.linalg.norm(pos - gp_))
                signed_d = f" sd={sd_:+.3f} dg={dg_:.2f}"
            # Route anchor diagnostic: did the best route's deterministic
            # rollout pass collision checks?
            anchor_m = getattr(self, "_last_route_anchor_margin", None)
            anchor_ok = "?" if anchor_m is None else ("OK" if anchor_m > 0.01 else f"BAD({anchor_m:.3f})")
            n_safe = int(np.sum(min_margin > 0.0)) if min_margin is not None else 0
            gates_passed = (
                self._n_gates if self._target_gate < 0 else int(self._target_gate)
            )
            print(
                f"[SPATIAL-SCENARIO-MPC] step={self._tick:04d} gate={self._target_gate} "
                f"passed={gates_passed}/{self._n_gates} "
                f"v={speed:.2f} pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]{signed_d} "
                f"rpy=[{rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}] "
                f"margin={best_margin:.3f} mode={selection_label} "
                f"route={anchor_ok} safe={n_safe}/{len(cost)} "
                f"plan={self._last_plan_ms:.1f}ms (ema {self._plan_ms_ema:.1f}) "
                f"u=[{u0[0]:+.3f},{u0[1]:+.3f},{u0[2]:+.3f},{u0[3]:.4f}]"
            )

        return self._finish_action(obs, pos, u0)

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _print_episode_summary(
        self, new_target: int, terminated: bool, truncated: bool,
    ) -> None:
        """One-line summary printed once at episode end (always, regardless
        of DEBUG_PRINT_ENABLED) so every run leaves a clear trace in the log.
        """
        if self._summary_printed:
            return
        self._summary_printed = True

        if new_target < 0:
            outcome = "FINISHED"
            gates_passed = self._n_gates
        elif truncated:
            outcome = "TIMEOUT"
            gates_passed = max(0, int(self._target_gate))
        else:
            outcome = "CRASH"
            gates_passed = max(0, int(self._target_gate))

        flight_time = self._tick * self._dt
        print(
            f"[SPATIAL-EP] {outcome:<8s} gates={gates_passed}/{self._n_gates} "
            f"time={flight_time:5.2f}s ticks={self._tick:4d} "
            f"plans={self._plan_counter:4d} plan_ms_ema={self._plan_ms_ema:5.1f}"
        )

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        new_target = int(obs["target_gate"])
        gate_changed = new_target != self._target_gate

        # A gate was just passed: remember its index so the MPPI cost keeps
        # penalising the drone for grazing its frame on the way out.
        if (
            gate_changed
            and 0 <= self._target_gate < self._n_gates
            and new_target > self._target_gate
        ):
            self._prev_target_gate = self._target_gate

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
            self._print_episode_summary(new_target, terminated, truncated)
            self._finished = True
            return True

        # --- Discover gate / obstacle positions (sensor range) ---
        any_gate_changed = False
        gate_deltas: list[tuple[int, np.ndarray]] = []  # (index, delta_pos)
        for i in range(self._n_gates):
            if obs["gates_visited"][i] and not self._gates_visited[i]:
                self._gates_visited[i] = True
                old_pos = self._gate_positions[i].copy()
                self._gate_positions[i] = np.array(obs["gates_pos"][i], dtype=np.float64)
                self._gate_quats[i] = np.array(obs["gates_quat"][i], dtype=np.float64)
                self._gate_rotmats[i] = Rot.from_quat(self._gate_quats[i]).as_matrix()
                delta_vec = self._gate_positions[i] - old_pos
                delta = float(np.linalg.norm(delta_vec))
                if delta > 0.01:
                    any_gate_changed = True
                    gate_deltas.append((i, delta_vec))
                    print(
                        f"[SPATIAL] Gate {i} discovered: "
                        f"[{old_pos[0]:.2f},{old_pos[1]:.2f},{old_pos[2]:.2f}] -> "
                        f"[{self._gate_positions[i][0]:.2f},{self._gate_positions[i][1]:.2f},"
                        f"{self._gate_positions[i][2]:.2f}] (d={delta:.3f}m)"
                    )

        any_obs_changed = False
        obs_deltas: list[tuple[int, np.ndarray]] = []  # (index, delta_pos)
        for i in range(len(self._obstacle_positions)):
            if obs["obstacles_visited"][i] and not self._obstacles_visited[i]:
                self._obstacles_visited[i] = True
                old_pos = self._obstacle_positions[i].copy()
                self._obstacle_positions[i] = np.array(obs["obstacles_pos"][i], dtype=np.float64)
                delta_vec = self._obstacle_positions[i] - old_pos
                delta = float(np.linalg.norm(delta_vec))
                if delta > 0.01:
                    any_obs_changed = True
                    obs_deltas.append((i, delta_vec))
                    print(
                        f"[SPATIAL] Obstacle {i} discovered: "
                        f"[{old_pos[0]:.2f},{old_pos[1]:.2f},{old_pos[2]:.2f}] -> "
                        f"[{self._obstacle_positions[i][0]:.2f},{self._obstacle_positions[i][1]:.2f},"
                        f"{self._obstacle_positions[i][2]:.2f}] (d={delta:.3f}m)"
                    )

        # Rebuild spatial geometry and force replan on discovery
        if any_gate_changed or any_obs_changed:
            self._rebuild_geometry()

            # --- Smooth-warp the cached trajectory instead of discarding ---
            # The old trajectory is ~95% correct; we just shift it toward the
            # newly-discovered positions so MPPI has a great warm-start.
            warped_seq = self._warp_cached_sequence_for_discovery(
                gate_deltas, obs_deltas,
            )
            if warped_seq is not None:
                self._cached_u_sequence = warped_seq
                self._mean_u = warped_seq.copy()
                # Store for injection into the next sample_sequences call
                self._warped_discovery_seq = warped_seq.copy()
                # Keep the old trajectories as warm-start hints (don't discard)
                # _last_best_traj / _last_route_anchor_traj stay as-is:
                # they'll be outcompeted naturally if the warped plan is worse.
            else:
                self._active_route_points = None
                self._last_best_traj = None
                self._last_route_anchor_traj = None

            self._sigma = np.maximum(self._sigma, self.SIGMA_INIT[None, :])
            self._last_planner_tick = -10**9
            self._cached_plan_tick = -10**9
            self._cached_u_index = 0

        return self._finished

    # ------------------------------------------------------------------
    #  Trajectory warping on object discovery
    # ------------------------------------------------------------------
    def _warp_cached_sequence_for_discovery(
        self,
        gate_deltas: list[tuple[int, np.ndarray]],
        obs_deltas: list[tuple[int, np.ndarray]],
    ) -> np.ndarray | None:
        """Smoothly warp the cached control sequence to account for discovered
        gate/obstacle position changes.

        For each discovered object with position delta, we:
        1. Forward-simulate the old mean to get the predicted trajectory
        2. For each trajectory point, compute a smooth shift weight based on
           proximity to the changed object (Gaussian falloff)
        3. Convert the required position correction into attitude+thrust
           perturbations on the control sequence

        Returns the warped control sequence, or None if no valid cached data.
        """
        if (self._cached_u_sequence is None or
                len(gate_deltas) == 0 and len(obs_deltas) == 0):
            return None

        # We need a predicted trajectory from the old mean sequence
        old_seq = self._cached_u_sequence.copy()
        pos0 = self._last_pos.copy()
        vel0 = np.zeros(3, dtype=np.float64)  # approximate
        rpy0 = self._last_rpy.copy()
        drpy0 = self._last_drpy.copy()

        # Quick single-sample rollout of the old sequence
        old_traj, _, _, _ = self._rollout_rpy_thrust(
            old_seq[None, :, :], pos0, vel0, rpy0, drpy0,
        )
        old_traj = old_traj[0]  # (N+1, 3)

        N = self.MPC_N
        # Accumulate desired position corrections at each horizon step
        pos_correction = np.zeros((N + 1, 3), dtype=np.float64)

        GATE_INFLUENCE_RADIUS = 1.5   # how far the shift reaches [m]
        OBS_INFLUENCE_RADIUS = 0.8

        for gi, delta in gate_deltas:
            gate_pos = self._gate_positions[gi]  # already updated
            for k in range(N + 1):
                dist = float(np.linalg.norm(old_traj[k] - (gate_pos - delta)))
                w = math.exp(-0.5 * (dist / GATE_INFLUENCE_RADIUS) ** 2)
                pos_correction[k] += w * delta

        for oi, delta in obs_deltas:
            obs_pos = self._obstacle_positions[oi]  # already updated
            for k in range(N + 1):
                dist = float(np.linalg.norm(old_traj[k] - (obs_pos - delta)))
                if dist < _EPS:
                    continue
                # Push away from obstacle's new position
                w = math.exp(-0.5 * (dist / OBS_INFLUENCE_RADIUS) ** 2)
                away_dir = old_traj[k] - obs_pos
                away_norm = float(np.linalg.norm(away_dir))
                if away_norm > _EPS:
                    # Add a small push in the away direction scaled by delta magnitude
                    push_mag = w * float(np.linalg.norm(delta))
                    pos_correction[k] += push_mag * away_dir / away_norm

        # Convert position corrections to control perturbations
        # Use small-angle approximation: dx ~ dt*(dv), dv ~ dt*(da)
        # da_correction ~ pos_correction / (dt^2), then roll~-ay/g, pitch~ax/g
        warped = old_seq.copy()
        dt = self.MPC_DT
        for k in range(N):
            # Desired acceleration correction to shift the trajectory
            # pos_correction[k+1] ≈ 0.5 * a_corr * dt^2 (leading term)
            a_corr = 2.0 * pos_correction[k + 1] / max(dt * dt, _EPS)
            # Clamp to reasonable magnitude (max ~2 m/s^2 correction)
            a_mag = float(np.linalg.norm(a_corr))
            if a_mag > 2.0:
                a_corr = a_corr * (2.0 / a_mag)

            # Convert to attitude perturbation
            d_roll = float(np.clip(-a_corr[1] / self._g, -0.12, 0.12))
            d_pitch = float(np.clip(a_corr[0] / self._g, -0.12, 0.12))
            d_thrust = float(np.clip(
                a_corr[2] * self._mass_estimate / self._cmd_f_coef,
                -0.02, 0.02,
            ))

            warped[k, 0] += d_roll
            warped[k, 1] += d_pitch
            warped[k, 3] += d_thrust

        return self._clip_u(warped)

    def render_callback(self, sim: "Sim"):
        from crazyflow.sim.visualize import draw_line, draw_points

        drone_pos = getattr(self, "_last_pos", np.zeros(3, dtype=np.float64))

        if self._active_route_points is not None and len(self._active_route_points) > 1:
            draw_line(sim, self._active_route_points, rgba=(1.0, 0.85, 0.1, 0.90),
                      start_size=2.2, end_size=2.2)

        if self._last_best_traj is not None and len(self._last_best_traj) > 1:
            draw_line(sim, self._last_best_traj, rgba=(0.0, 1.0, 0.2, 0.85),
                      start_size=2.0, end_size=2.0)

        if self._last_route_anchor_traj is not None and len(self._last_route_anchor_traj) > 1:
            draw_line(sim, self._last_route_anchor_traj, rgba=(0.0, 0.9, 1.0, 0.50),
                      start_size=1.5, end_size=1.5)

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            draw_points(sim, gp.reshape(1, -1), rgba=(1.0, 1.0, 0.0, 1.0), size=0.04)
