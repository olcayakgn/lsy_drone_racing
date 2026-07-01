"""Spatial Scenario MPC Controller — MPPI planner with full RPY+thrust dynamics.

Built on a spatial curvilinear (Bishop-frame) coordinate system, it combines the
stochastic scenario / MPPI planning framework with:
  - Full rotational dynamics model (roll, pitch, yaw + angular rates + thrust)
  - Spatial curvilinear coordinate system via Parallel Transport (Bishop) frame
  - Dynamic flight corridor constraints projected onto the transverse plane
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
import traceback
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as Rot

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

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

    def draw_line(*a: object, **kw: object):
        """No-op stand-in so render code keeps working when spatial geometry is unavailable."""


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

    # ==========================================================================
    #  GEOMETRY / PHYSICAL PARAMETERS
    # ==========================================================================
    OBSTACLE_RADIUS = 0.15  # collision cylinder radius around obstacle poles [m]
    OBSTACLE_BUFFER = 0.08  # soft penalty zone beyond obstacle radius [m]
    DRONE_RADIUS = 0.05  # Crazyflie half-diagonal + margin [m]

    GATE_HALF_OPENING = 0.20  # inner edge half-width of gate opening [m]
    GATE_OUTER_HALF = 0.28  # outer edge half-width of gate frame [m]
    GATE_POST_RADIUS = 0.0  # box surface distance for gate posts [m]
    GATE_FRAME_BUFFER = 0.12  # soft penalty zone around gate frame surfaces [m]
    GATE_CLEARANCE = 0.045  # min clearance margin for valid gate crossing [m]
    GATE_PLANE_SLAB = 0.16  # gate-plane slab half-thickness for slab penalty [m]

    FUNNEL_LENGTH = 0.85  # approach funnel length — guides drone onto gate centreline [m]
    FUNNEL_OUTER_HALF = 0.20  # funnel half-width at gate plane — matches opening [m]
    APPROACH_DIST = 0.45  # approach waypoint distance before gate [m]
    EXIT_DIST = 0.35  # exit waypoint distance after gate [m]
    ALIGN_START_DIST = 1.20  # distance to gate where velocity alignment begins [m]
    # max dist at which W_NOT_CROSSED_REACHABLE activates (caps horizon-based threshold) [m]
    REACHABLE_DIST_CAP = 3.0
    GATE_CROSS_DEPTH = 0.05  # terminal clearance past gate plane for zero crossing penalty [m]

    GROUND_CLEARANCE = 0.10  # minimum altitude above ground [m]
    CEILING = 1.80  # maximum altitude [m]

    # ==========================================================================
    #  SPEED LIMITS
    # ==========================================================================
    V_CRUISE = 1.75  # nominal cruise speed [m/s]
    V_GATE = 1.4  # reduced speed near gate crossing [m/s]
    V_MAX = 2.80  # absolute speed limit [m/s]

    # ==========================================================================
    #  ATTITUDE / ACTUATOR LIMITS
    # ==========================================================================
    MAX_ROLL_PITCH_CMD = 0.55  # max roll/pitch command [rad]
    MAX_YAW_CMD = 0.50  # max yaw command magnitude [rad]

    # ==========================================================================
    #  MPPI SAMPLING / HORIZON
    # ==========================================================================
    MPC_N = 9  # prediction horizon steps (9 × 0.11 = 0.99s)
    MPC_DT = 0.110  # prediction time step [s]
    K_SAMPLES = 4096  # number of MPPI rollout samples per plan (JAX-parallelised)
    USE_JAX = True  # use JAX       JIT+vmap acceleration (NumPy fallback if unavailable)
    N_ELITES = 1  # number of elite samples for distribution update
    TEMPERATURE = 50.0  # MPPI softmax temperature (lower = greedier)
    NOISE_RHO = 0.84  # temporal correlation of MPPI noise (AR(1) coefficient)
    CMD_FILTER_ALPHA = 0.86  # low-pass filter on output command (0=old, 1=new)
    RISK_BLEND_MARGIN = 0.07  # margin threshold for risk-aware blending [m]
    PLANNER_HZ = 50.0  # replanning frequency [Hz]
    RENDER_EVERY = 1  # render every Nth step
    DRAW_FULL_DEBUG_GEOMETRY = False  # draw all debug geometry in render
    EVALUATE_MEAN_ROLLOUT = True  # also evaluate the mean sequence as a sample
    ROLLOUT_SUB_STEPS = 2  # Euler sub-steps per MPC interval for numerical stability

    # ==========================================================================
    #  CANDIDATE SELECTION
    # ==========================================================================
    SAFE_SELECTION_MARGIN = 0.020  # min margin for a candidate to be considered safe [m]
    SELECTION_MARGIN = 0.075  # preferred margin for robust selection [m]
    ROBUST_CROSS_RADIUS = 0.090  # max gate-local YZ radius for a "center" crossing [m]
    LOOSE_CROSS_RADIUS = 0.125  # max gate-local YZ radius for a "loose" crossing [m]
    ELITE_BLEND_COUNT = 1  # number of elite candidates to blend
    ELITE_TEMPERATURE = 18.0  # softmax temperature for elite blending

    TRACK_LOOKAHEAD = 0.060  # time lookahead for cached-command interpolation [s]

    # ==========================================================================
    #  DEBUG / DIAGNOSTICS
    # ==========================================================================
    DEBUG_PRINT_ENABLED = True  # per-step diagnostic prints (episode summaries always print)
    DEBUG_EVERY_PLANS = 10  # print diagnostics every Nth planning step

    FORGET_OLD_SCENARIOS = True  # reset MPPI distribution each plan (vs. warm-start)
    USE_PREVIOUS_ROUTE_CANDIDATE = False  # inject previous route into candidates
    RESET_SIGMA_EACH_PLAN = True  # reset exploration noise each plan
    COMMAND_ONLY_NONCRASHING = True  # only select non-crashing samples for output

    # ==========================================================================
    #  MPPI NOISE DISTRIBUTION (4-D: [roll_cmd, pitch_cmd, yaw_cmd, thrust])
    # ==========================================================================
    SIGMA_INIT = np.array([0.35, 0.35, 0.25, 0.120], dtype=np.float64)  # initial exploration std
    SIGMA_MIN = np.array([0.06, 0.06, 0.04, 0.020], dtype=np.float64)  # minimum exploration std
    SIGMA_MAX = np.array([0.60, 0.60, 0.45, 0.200], dtype=np.float64)  # maximum exploration std

    # ==========================================================================
    #  ROUTE PLANNING
    # ==========================================================================
    ROUTE_PRESELECT_K = 4  # number of route candidates to keep after scoring
    ROUTE_TOP_K = 2  # number of top routes to keep after rollout ranking
    ROUTE_MEAN_BLEND = 1.00  # blend factor for route mean into MPPI mean (0=keep, 1=replace)
    ROUTE_COMMAND_BLEND = 0.24  # blend factor of route anchor into final command
    ROUTE_SAMPLE_SPACING = 0.18  # polyline sampling spacing for route evaluation [m]
    ROUTE_OBS_LOOKAHEAD_MARGIN = 0.55  # obstacle lookahead margin for route detours [m]
    ROUTE_SIDE_OFFSETS = (0.0, 0.35, -0.35, 0.65, -0.65)  # lateral offsets for route variants [m]
    ROUTE_ENTRY_Y_OFFSETS = (0.0, 0.08, -0.08)  # gate-local Y offsets for entry variants [m]
    ROUTE_ENTRY_Z_OFFSETS = (0.0, 0.06, -0.06)  # gate-local Z offsets for entry variants [m]
    ROUTE_GATE_SAFE_DIST = 0.18  # min distance from route to any gate frame bar [m]
    ROUTE_GATE_CHECK_SPACING = 0.12  # sample spacing for gate proximity checks [m]

    # ==========================================================================
    #  COST WEIGHTS — REFERENCE TRACKING
    # ==========================================================================
    W_REF_POS = 11.0  # position tracking error weight
    W_REF_VEL = 3.0  # velocity tracking error weight
    W_TERMINAL_REF = 40.0  # terminal position error weight
    W_PROGRESS = 26.0  # forward progress toward gate reward weight
    W_GATE_DISTANCE_STAGE = 5.5  # per-step distance-to-gate penalty weight
    W_GATE_DISTANCE_TERMINAL = 42.0  # terminal distance-to-gate penalty weight
    W_GATE_CLOSING = 28.0  # reward for reducing distance to gate per step
    W_GATE_APPROACH_VEL = 18.0  # penalty for retreating from gate within 0.30m (not-yet-crossed)
    W_GATE_ALTITUDE = 15.0  # penalty for being below target gate altitude (climb incentive)
    W_NOT_CROSSED_REACHABLE = 1800.0  # penalty for not crossing a reachable gate

    # ==========================================================================
    #  COST WEIGHTS — GATE CROSSING QUALITY
    # ==========================================================================
    W_CROSS_CENTER = 4500.0  # reward for crossing gate near center
    W_BAD_CROSS = 30000.0  # penalty for crossing outside safe opening
    BONUS_GOOD_CROSS = 2100.0  # flat bonus for a clean center crossing

    # ==========================================================================
    #  COST WEIGHTS — OBSTACLE AVOIDANCE
    # ==========================================================================
    W_POLE_BUFFER = 2600.0  # soft buffer zone penalty around poles
    W_POLE_COLLISION = 25000.0  # hard collision penalty for poles
    W_POLE_NEAR_EXP = 15.0  # exponential proximity penalty for poles

    # ==========================================================================
    #  COST WEIGHTS — GATE FRAME AVOIDANCE
    # ==========================================================================
    W_FRAME_BUFFER = 10000.0  # soft buffer zone penalty around gate frame bars
    W_FRAME_COLLISION = 100000.0  # hard collision penalty for gate frame bars
    W_FRAME_SLAB = 35000.0  # penalty for being in gate slab outside opening
    W_FUNNEL = 2800.0  # funnel centering penalty (guides approach)

    # Cost multipliers for gate frame avoidance at different gate offsets.
    # 1.0 = full cost; lower values reduce influence of non-target gates.
    GATE_COST_MULT_CURRENT = 1.00  # current target gate
    GATE_COST_MULT_CURRENT_PASSED = 0.85  # current gate after crossing / exit zone
    GATE_COST_MULT_NEXT_ACTIVE = 0.85  # next gate when actively approaching
    GATE_COST_MULT_NEXT_PASSIVE = 0.70  # next gate when not yet approaching
    GATE_COST_MULT_NEXT2 = 0.55  # gate two ahead (horizon can reach it)
    GATE_COST_MULT_PAST = 0.50  # all previously passed gates (U-turn safety)

    # ==========================================================================
    #  COST WEIGHTS — SAFETY / LIMITS
    # ==========================================================================
    W_ALTITUDE = 6000.0  # soft altitude boundary penalty
    W_ALTITUDE_HARD = 40000.0  # hard altitude violation penalty (floor/ceiling)
    W_SPEED_LIMIT = 18.0  # overspeed penalty weight
    W_INPUT = 0.025  # control magnitude regularisation
    W_DINPUT = 0.075  # control rate-of-change regularisation
    W_LATERAL = 0.020  # lateral deviation penalty

    # ==========================================================================
    #  COST WEIGHTS — SPATIAL / ROTATIONAL (Bishop frame)
    # ==========================================================================
    W_CORRIDOR = 800.0  # soft corridor boundary penalty
    W_CORRIDOR_HARD = 4000.0  # hard corridor violation penalty
    W_SPATIAL_PROGRESS = 4.0  # reward for longitudinal progress along path
    W_CURVATURE_SPEED = 2.0  # penalty for exceeding curvature-adapted speed limit
    W_ATTITUDE_SMOOTH = 0.15  # angular rate smoothness penalty
    W_ATTITUDE_LIMIT = 400.0  # penalty for roll/pitch beyond safe limit

    # ==========================================================================
    #  ROUTE TRACKING FEEDBACK (closed-loop PD correction)
    # ==========================================================================
    USE_PROJECTED_ROUTE_TRACKING = True  # use polyline projection vs. time-indexed tracking
    PATH_LOOKAHEAD_DIST = 0.32  # base lookahead distance for path projection [m]
    PATH_LOOKAHEAD_SPEED_GAIN = 0.08  # lookahead increase per m/s of speed
    PATH_LOOKAHEAD_MAX = 0.58  # maximum lookahead distance [m]
    CROSS_TRACK_KP = 5.20  # cross-track proportional gain
    CROSS_TRACK_KD = 3.10  # cross-track derivative gain
    ALONG_TRACK_KP = 0.80  # along-track proportional gain
    ALONG_TRACK_KD = 0.85  # along-track derivative gain
    TURN_BRAKE_GAIN = 1.35  # braking gain at sharp turns
    TURN_SPEED_MIN = 0.95  # minimum speed at sharpest turns [m/s]
    TURN_SPEED_MAX = 1.60  # maximum speed in turns [m/s]
    TURN_ANGLE_FOR_SLOWDOWN = 0.55  # turn angle threshold for speed reduction [rad]
    CROSS_TRACK_SLOWDOWN_START = 0.12  # cross-track error where slowdown begins [m]
    CROSS_TRACK_SLOWDOWN_FULL = 0.34  # cross-track error where slowdown is full [m]

    # ==========================================================================
    #  REACTIVE OBSTACLE PUSH (APF — artificial potential field)
    # ==========================================================================
    APF_INFLUENCE = 0.36  # influence radius for reactive obstacle push [m]
    APF_GAIN = 0.22  # repulsion gain
    APF_MAX = 0.65  # maximum repulsion acceleration [m/s²]

    # ==========================================================================
    #  LAUNCH ALTITUDE HOLD
    # ==========================================================================
    LAUNCH_HOLD_TIME = 0.25  # hold initial altitude for this duration [s]
    LAUNCH_BLEND_TIME = 0.70  # blend to reference altitude over this duration [s]
    VZ_REF_MAX = 0.65  # maximum vertical reference velocity [m/s]

    # ==========================================================================
    #  DISCOVERY WARPING (smooth trajectory shift on gate/obstacle discovery)
    # ==========================================================================
    WARP_GATE_INFLUENCE_RADIUS = 1.5  # Gaussian falloff radius for gate discovery shift [m]
    WARP_OBS_INFLUENCE_RADIUS = 0.8  # Gaussian falloff radius for obstacle discovery shift [m]

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(self, obs: dict[str, "NDArray[np.floating]"], info: dict, config: dict):
        """Initialise drone parameters, gate/obstacle state, and the MPPI planner.

        Args:
            obs: Initial environment observation (positions, quaternions, visited flags).
            info: Additional environment info dict.
            config: Race configuration (env frequency, sim physics, drone model, seed).
        """
        super().__init__(obs, info, config)

        self._g = 9.81
        self._dt = 1.0 / float(config.env.freq)

        # --- drone physical parameters ---
        drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
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
        self._gate_positions = np.array([g.tolist() for g in obs["gates_pos"]], dtype=np.float64)
        self._gate_quats = np.array([g.tolist() for g in obs["gates_quat"]], dtype=np.float64)
        self._gate_rotmats = [Rot.from_quat(q).as_matrix() for q in self._gate_quats]
        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])

        self._obstacle_positions = np.array(
            [p.tolist() for p in obs["obstacles_pos"]], dtype=np.float64
        )
        self._n_obstacles = len(self._obstacle_positions)
        self._gates_visited = obs["gates_visited"].copy()

        # List of all passed gate indices. Their frames stay in the MPPI
        # cost so the drone avoids them on U-turns and dips (not just the
        # most recently passed one).
        self._past_gates = []
        print(f"[SPATIAL-INIT] drone_pos={obs['pos']}")
        for i in range(self._n_gates):
            print(
                f"SPATIAL-INIT] gate{i}: pos={self._gate_positions[i]} "
                f"visited={self._gates_visited[i]}"
            )
        self._obstacles_visited = np.array(
            obs.get("obstacles_visited", np.zeros(len(self._obstacle_positions), dtype=bool)),
            dtype=bool,
        )

        # --- spatial geometry engine (Bishop frame + corridors) ---
        self._geo = None
        self._prev_s = 0.0
        if _HAS_SPATIAL:
            self._init_geometry_engine()

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
            1, int(round(1.0 / max(self.PLANNER_HZ * self._dt, _EPS)))
        )
        self._last_planner_tick = -(10**9)
        self._cached_plan_tick = -(10**9)
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

        # --- real-time lag detection ---
        self._compute_deadline_s = self._dt
        self._compute_ms_ema = 0.0
        self._compute_lag_count = 0
        self._compute_total_count = 0

        print(
            f"[SPATIAL-SCENARIO-MPC] ready: K={self.K_SAMPLES}, N={self.MPC_N}, "
            f"dt={self.MPC_DT:.3f}s, horizon={self.MPC_N * self.MPC_DT:.2f}s, "
            f"planner_hz={self.PLANNER_HZ:.1f}, "
            f"physics={self._physics}, "
            f"rotor_lag={'YES' if self._has_rotor_lag else 'NO'}, "
            f"drag={'YES' if self._has_drag else 'NO'}, "
            f"spatial={'YES' if self._geo is not None else 'NO'}"
        )

        # --- JAX-accelerated evaluator (built after all state is ready) ---
        self._jax_eval = None
        self._try_init_jax()

    # ------------------------------------------------------------------
    #  Spatial geometry initialisation
    # ------------------------------------------------------------------
    def _init_geometry_engine(self):
        """Initialise the Bishop-frame spatial geometry engine from gate/obstacle layout."""
        self._rebuild_geometry()

    def _rebuild_geometry(self):
        """(Re)build the GeometryEngine after gate/obstacle discovery updates."""
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
        except (ValueError, RuntimeError) as e:
            # Catch expected GeometryEngine construction errors (avoids broad-except).
            print(f"[SPATIAL] GeometryEngine build failed: {e}")
            traceback.print_exc()
            self._geo = None

    # ------------------------------------------------------------------
    #  Cartesian ↔ Spatial conversions
    # ------------------------------------------------------------------
    def _cartesian_to_spatial(
        self, pos: np.ndarray, vel: np.ndarray, rpy: np.ndarray, drpy: np.ndarray
    ) -> np.ndarray | None:
        """Convert world-frame state to spatial curvilinear (s, w1, w2) via Bishop frame."""
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
        """Penalise lateral deviations outside the spatial flight corridor(batch of K positions)."""
        if self._geo is None:
            return np.zeros(positions.shape[0], dtype=np.float64)

        K = positions.shape[0]
        cost = np.zeros(K, dtype=np.float64)

        # Scale stride with K: for K=100 use stride≈4, for K=4096 use stride≈128
        stride = max(4, K // 32)
        for i in range(0, K, stride):
            s = self._geo.get_closest_s(positions[i], s_guess=self._prev_s)
            f = self._geo.get_frame(s)
            r_vec = positions[i] - f["pos"]
            w1 = float(np.dot(r_vec, f["n1"]))

            lb, ub = self._geo.get_static_bounds(s)

            viol_lo = max(lb - w1, 0.0)
            viol_hi = max(w1 - ub, 0.0)
            viol = viol_lo + viol_hi
            c = self.W_CORRIDOR * viol**2

            deep = max(viol - 0.10, 0.0)
            c += self.W_CORRIDOR_HARD * deep**2

            # Apply to this sample and its neighbors
            end = min(i + stride, K)
            cost[i:end] = c

        return cost

    def _spatial_progress_and_curvature(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-sample progress reward and curvature-adapted speed penalty."""
        if self._geo is None:
            return np.zeros(positions.shape[0]), np.zeros(positions.shape[0])

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
        curv_cost = overspeed**2

        return progress, curv_cost

    # ------------------------------------------------------------------
    #  Full RPY + thrust dynamics rollout (vectorised)
    # ------------------------------------------------------------------

    def _rollout_rpy_thrust(
        self,
        samples: np.ndarray,
        pos0: np.ndarray,
        vel0: np.ndarray,
        rpy0: np.ndarray,
        drpy0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward-integrate K samples with full RPY+thrust dynamics (vectorised)."""
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

        c_rpy = self._rpy_coef  # (3,)
        c_drpy = self._rpy_rates_coef  # (3,)
        c_cmd = self._cmd_rpy_coef  # (3,)
        g_z = -self._g
        inv_mass = 1.0 / self._mass_estimate
        acc_coef = self._acc_coef
        cmd_f = self._cmd_f_coef

        # Rotor lag state
        has_rotor_lag = self._has_rotor_lag
        if has_rotor_lag:
            t_act = np.full((K, 1), self._thrust_actual, dtype=np.float64)
            sub_dt_inv_tau = sub_dt / self._thrust_time_coef

        # Drag coefficients
        has_drag = self._has_drag
        if has_drag:
            drag_xy = self._drag_xy
            drag_z_coef = self._drag_z

        for k in range(N):
            u = samples[:, k, :]  # (K, 4)
            cmd_rpy_k = u[:, :3]  # (K, 3) roll/pitch/yaw commands
            t_cmd = u[:, 3:4]  # (K, 1) thrust command

            for _ in range(n_sub):
                # --- Rotor lag: first-order thrust response ---
                if has_rotor_lag:
                    t_act = t_act + (t_cmd - t_act) * sub_dt_inv_tau
                    t_phys = acc_coef + cmd_f * t_act
                else:
                    t_phys = acc_coef + cmd_f * t_cmd

                # --- Rotational dynamics (sub-step) ---
                ddrpy = c_rpy[None, :] * rpy + c_drpy[None, :] * drpy + c_cmd[None, :] * cmd_rpy_k

                # --- Rotation matrix elements ---
                phi = rpy[:, 0:1]
                theta = rpy[:, 1:2]
                psi = rpy[:, 2:3]

                cx, sx = np.cos(phi), np.sin(phi)
                cy, sy = np.cos(theta), np.sin(theta)
                cz, sz = np.cos(psi), np.sin(psi)

                # Third column of R_IB (body Z in world) — for thrust
                r_02 = sx * sz + cx * cz * sy  # (K, 1)
                r_12 = cx * sy * sz - cz * sx  # (K, 1)
                r_22 = cx * cy  # (K, 1)

                # --- World-frame acceleration ---
                acc = np.empty_like(vel)
                acc[:, 0:1] = r_02 * t_phys * inv_mass
                acc[:, 1:2] = r_12 * t_phys * inv_mass
                acc[:, 2:3] = g_z + r_22 * t_phys * inv_mass

                # --- Aerodynamic drag (body-frame linear) ---
                if has_drag:
                    # Full rotation matrix columns 0,1 for drag transform
                    r_00 = cz * cy
                    r_01 = cz * sy * sx - sz * cx
                    r_10 = sz * cy
                    r_11 = sz * sy * sx + cz * cx
                    r_20 = -sy
                    r_21 = cy * sx
                    # World velocity → body frame: v_body = R^T @ v_world
                    vx, vy, vz_w = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]
                    vb_x = r_00 * vx + r_10 * vy + r_20 * vz_w
                    vb_y = r_01 * vx + r_11 * vy + r_21 * vz_w
                    vb_z = r_02 * vx + r_12 * vy + r_22 * vz_w
                    # Drag force in body frame (coefficients are negative → opposes motion)
                    fd_x = drag_xy * vb_x
                    fd_y = drag_xy * vb_y
                    fd_z = drag_z_coef * vb_z
                    # Transform back to world: a_drag = R @ F_body / m
                    acc[:, 0:1] += (r_00 * fd_x + r_01 * fd_y + r_02 * fd_z) * inv_mass
                    acc[:, 1:2] += (r_10 * fd_x + r_11 * fd_y + r_12 * fd_z) * inv_mass
                    acc[:, 2:3] += (r_20 * fd_x + r_21 * fd_y + r_22 * fd_z) * inv_mass

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
        """Return gate normal vector pointing from drone toward the gate."""
        if gi < 0 or gi >= self._n_gates:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        normal = self._gate_rotmats[gi][:, 0].copy()
        to_gate = self._gate_positions[gi] - from_pos
        if float(np.dot(to_gate, normal)) < 0.0:
            normal = -normal
        n = np.linalg.norm(normal)
        return normal / max(float(n), _EPS)

    @staticmethod
    def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        """Normalise vector to unit length; return fallback if near-zero."""
        n = float(np.linalg.norm(v))
        if n > _EPS:
            return np.asarray(v, dtype=np.float64) / n
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).copy()

    @staticmethod
    def _world_to_gate_local(points: np.ndarray, gp: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Transform world-frame points into gate-local coordinates."""
        return (points - gp) @ R

    @staticmethod
    def _point_segment_distance_xy(
        point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """2D point-to-segment distance (used for obstacle detour planning)."""
        ab = b_xy - a_xy
        denom = float(np.dot(ab, ab))
        if denom < _EPS:
            return float(np.linalg.norm(point_xy - a_xy)), 0.0, a_xy.copy()
        t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
        closest = a_xy + t * ab
        return float(np.linalg.norm(point_xy - closest)), t, closest

    def _apply_launch_altitude_ramp(self, ref_pos: np.ndarray, ref_vel: np.ndarray) -> None:
        """Clamp reference trajectory to a safe altitude ramp during takeoff."""
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

    def _distance_to_gate_bars(self, local: np.ndarray) -> np.ndarray:
        """Unsigned distance from points (gate-local coords) to nearest gate frame collision box."""
        K = local.shape[0]
        x = local[:, 0]
        y = local[:, 1]
        z = local[:, 2]

        # Box definitions: (center_y, center_z, half_y, half_z)
        # All boxes have half_x = 0.01 (gate plane thickness)
        hx = 0.01
        boxes = [
            (0.0, 0.28, 0.36, 0.08),  # top
            (0.0, -0.28, 0.36, 0.08),  # bottom
            (-0.28, 0.0, 0.08, 0.36),  # left
            (0.28, 0.0, 0.08, 0.36),  # right
        ]

        d_min = np.full(K, 1e6, dtype=np.float64)
        for cy, cz, hy, hz in boxes:
            # Signed distance components for an axis-aligned box
            dx = np.abs(x) - hx
            dy = np.abs(y - cy) - hy
            dz = np.abs(z - cz) - hz
            # Outside distance: Euclidean of positive components
            outside = np.sqrt(
                np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2
            )
            # Inside distance: max of negative components (closest wall)
            inside = np.maximum(dx, np.maximum(dy, dz))
            # Unsigned distance: 0 when inside, positive when outside
            dist = np.where(inside < 0.0, 0.0, outside)
            d_min = np.minimum(d_min, dist)

        return d_min

    def _gate_geometry_cost_and_margin(
        self, pos: np.ndarray, gp: np.ndarray, R: np.ndarray, active_funnel: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute frame collision + funnel + slab cost for one gate. Core gate avoidance cost."""
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
            cost += (
                0.80
                * self.W_FUNNEL
                * exit_fw
                * ((v_ey / exit_bound) ** 2 + (v_ez / exit_bound) ** 2)
            )

        margin_bar = d_bar_eff  # positive = clear, negative = inside box
        return cost, margin_bar

    # ------------------------------------------------------------------
    #  4-D control utilities
    # ------------------------------------------------------------------
    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        """Clip 4-D control vector [roll, pitch, yaw, thrust] to actuator limits."""
        out = np.array(u, dtype=np.float64, copy=True)
        out[..., 0] = np.clip(out[..., 0], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
        out[..., 1] = np.clip(out[..., 1], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
        out[..., 2] = np.clip(out[..., 2], -self.MAX_YAW_CMD, self.MAX_YAW_CMD)
        out[..., 3] = np.clip(out[..., 3], self._thrust_min, self._thrust_max)
        return out

    # ------------------------------------------------------------------
    #  Deterministic route candidates (Cartesian — reused from scenario MPC)
    # ------------------------------------------------------------------
    def _clip_route_point(self, point: np.ndarray) -> np.ndarray:
        """Clamp a single route waypoint to safe altitude bounds."""
        p = np.asarray(point, dtype=np.float64).copy()
        p[2] = float(np.clip(p[2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05))
        return p

    def _clean_route(self, points: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Deduplicate and altitude-clamp a list of waypoints into a clean polyline."""
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

    def _sample_polyline(self, route: np.ndarray, spacing: float | None = None) -> np.ndarray:
        """Resample a polyline route at uniform spacing for evaluation."""
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

    def _min_dist_to_gate_frame(self, point: np.ndarray, gate_idx: int) -> float:
        """Distance from a single world-space point to one gate's frame bars."""
        gp = self._gate_positions[gate_idx]
        R = self._gate_rotmats[gate_idx]
        local = (point - gp) @ R  # (3,)
        local_2d = local.reshape(1, 3)
        return float(self._distance_to_gate_bars(local_2d)[0])

    def _gate_frame_push_vector(self, point: np.ndarray, gate_idx: int) -> tuple[np.ndarray, float]:
        """World-space push direction away from nearest gate frame bar (for route deflection)."""
        gp = self._gate_positions[gate_idx]
        R = self._gate_rotmats[gate_idx]
        local = (point - gp) @ R  # (3,)

        # Find nearest bar center in gate-local coords
        hx = 0.01
        boxes = [
            (0.0, 0.28, 0.36, 0.08),  # top
            (0.0, -0.28, 0.36, 0.08),  # bottom
            (-0.28, 0.0, 0.08, 0.36),  # left
            (0.28, 0.0, 0.08, 0.36),  # right
        ]
        best_dist = 1e6
        best_push_local = np.array([0.0, 0.0, 0.0])
        for cy, cz, hy, hz in boxes:
            dx = abs(local[0]) - hx
            dy = abs(local[1] - cy) - hy
            dz = abs(local[2] - cz) - hz
            outside = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 + max(dz, 0.0) ** 2)
            inside = max(dx, dy, dz)
            dist = 0.0 if inside < 0.0 else outside
            if dist < best_dist:
                best_dist = dist
                # Push direction: from box center toward point, in gate-local frame
                push = np.array([local[0], local[1] - cy, local[2] - cz], dtype=np.float64)
                norm = float(np.linalg.norm(push))
                if norm > _EPS:
                    best_push_local = push / norm
                else:
                    best_push_local = np.array([0.0, -np.sign(cy + _EPS), -np.sign(cz + _EPS)])

        # Convert push direction back to world frame
        push_world = R @ best_push_local
        return push_world, best_dist

    def _batch_min_dist_to_gates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised min distance from N points to any gate frame bar across all gates."""
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

    def _deflect_route_from_gates(self, route: np.ndarray) -> np.ndarray:
        """Insert detour waypoints where route passes too close to gate frames."""
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
                    deflected[2] = float(
                        np.clip(deflected[2], self.GROUND_CLEARANCE + 0.03, self.CEILING - 0.05)
                    )
                    new_points.append(deflected)

            new_points.append(route[seg_i + 1].copy())

        return self._clean_route(new_points)

    def _straighten_near_gates(self, route: np.ndarray) -> np.ndarray:
        """Align waypoints near target gate onto the gate normal axis (prevents lateral drift)."""
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

    def _route_min_gate_clearance(self, route: np.ndarray) -> float:
        """Minimum clearance from sampled route points to any gate frame bar."""
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

    def _make_gate_tail(self, pos: np.ndarray) -> list[np.ndarray]:
        """Build approach→crossing→exit waypoint sequence for the target gate."""
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

    def _score_route(
        self, route: np.ndarray, pos: np.ndarray, vel: np.ndarray | None = None
    ) -> float:
        """Score a candidate route by length, alignment, obstacle/gate proximity, and smoothness."""
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
                    r_other = self._gate_rotmats[gi]
                    local = (cross_pt - gp_other) @ r_other
                    if (
                        abs(local[1]) < self.GATE_OUTER_HALF
                        and abs(local[2]) < self.GATE_OUTER_HALF
                    ):
                        score += 500.0  # heavy penalty for going through another gate
                    break  # one crossing per gate is enough

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            d0 = float(np.linalg.norm(pos - gp))
            d1 = float(np.linalg.norm(route[min(1, len(route) - 1)] - gp))
            score -= 12.0 * max(0.0, d0 - d1)

        return float(score)

    def _build_route_candidates(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate multiple candidate routes (direct, side, detour, reversal) and rank by score."""
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

        def add_route(points: list[np.ndarray]) -> None:
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
            if (
                0.02 < t_seg < 0.98
                and d_seg < self.OBSTACLE_RADIUS + self.ROUTE_OBS_LOOKAHEAD_MARGIN
            ):
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
                    add_route(
                        [
                            pos.copy(),
                            pre_blend,
                            detour,
                            post_blend,
                            entry_center,
                            approach,
                            gp.copy(),
                            exit_pt,
                        ]
                        + tail[3:]
                    )

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

                    add_route(
                        [
                            pos.copy(),
                            fwd_pt.copy(),
                            arc_pt,
                            mid_return,
                            entry_center,
                            approach,
                            gp.copy(),
                            exit_pt,
                        ]
                        + tail[3:]
                    )

        if not candidates:
            add_route([pos.copy()] + tail)

        # Deflect all candidates away from gate frames, then straighten near gates
        candidates = [
            self._straighten_near_gates(self._deflect_route_from_gates(r)) for r in candidates
        ]

        scored = [(self._score_route(r, pos, vel), r) for r in candidates]
        scored.sort(key=lambda item: item[0])
        top = scored[: max(1, int(self.ROUTE_PRESELECT_K))]
        return [r for _, r in top], [float(c) for c, _ in top]

    # ------------------------------------------------------------------
    #  Reference generation & deterministic PD sequence (now 4-D)
    # ------------------------------------------------------------------
    def _generate_references(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        route_points: np.ndarray | list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate position/velocity reference trajectory along the route for MPPI tracking."""
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
        for i, seg_len in enumerate(seg_lens):
            if i > 0 and seg_len > 0.02 and seg_lens[i - 1] > 0.02:
                d_prev = self._unit(route[i] - route[i - 1])
                d_next = self._unit(route[i + 1] - route[i])
                turn_cos = float(np.dot(d_prev, d_next))
                seg_turn[i] = max(0.0, 1.0 - turn_cos)  # 0=straight, 2=reversal
            if i + 1 < len(seg_lens) and seg_len > 0.02 and seg_lens[i + 1] > 0.02:
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

    def _make_pd_sequence(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        drpy: np.ndarray,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
    ) -> np.ndarray:
        """Deterministic PD warm-start: attitude+thrust sequence that tracks the reference."""
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
                a_des, target_yaw=0.0
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
                r_02 = sx * sz + cx * cz * sy
                r_12 = cx * sy * sz - cz * sx
                r_22 = cx * cy
                acc_w = np.array(
                    [
                        r_02 * thrust_phys * inv_m,
                        r_12 * thrust_phys * inv_m,
                        -self._g + r_22 * thrust_phys * inv_m,
                    ]
                )

                # Drag
                if self._has_drag:
                    r_00 = cz * cy
                    r_01 = cz * sy * sx - sz * cx
                    r_10 = sz * cy
                    r_11 = sz * sy * sx + cz * cx
                    r_20 = -sy
                    r_21 = cy * sx
                    vb_x = r_00 * v[0] + r_10 * v[1] + r_20 * v[2]
                    vb_y = r_01 * v[0] + r_11 * v[1] + r_21 * v[2]
                    vb_z = r_02 * v[0] + r_12 * v[1] + r_22 * v[2]
                    fd_x = self._drag_xy * vb_x
                    fd_y = self._drag_xy * vb_y
                    fd_z = self._drag_z * vb_z
                    acc_w[0] += (r_00 * fd_x + r_01 * fd_y + r_02 * fd_z) * inv_m
                    acc_w[1] += (r_10 * fd_x + r_11 * fd_y + r_12 * fd_z) * inv_m
                    acc_w[2] += (r_20 * fd_x + r_21 * fd_y + r_22 * fd_z) * inv_m

                v = v + acc_w * sub_dt
                p = p + v * sub_dt
                ddr = self._rpy_coef * r + self._rpy_rates_coef * dr + self._cmd_rpy_coef * cmd_arr
                dr = dr + ddr * sub_dt
                r = r + dr * sub_dt

        return self._clip_u(seq)

    def _accel_to_attitude_cmd(
        self, a_des: np.ndarray, target_yaw: float = 0.0
    ) -> tuple[float, float, float, float]:
        """Convert desired acceleration to [roll, pitch, yaw, thrust] commands (PD warm-start)."""
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
    def _try_init_jax(self):
        """Build JAX-compiled MPPI evaluator; falls back silently if unavailable."""
        if not self.USE_JAX or not _HAS_JAX:
            self._jax_eval = None
            return
        try:
            self._jax_eval = self._build_jax_fn()
            # Warm-up: trigger JIT compilation with the exact shapes used in planning.
            K, N = int(self.K_SAMPLES), int(self.MPC_N)
            P = int(self._n_gates)
            n_obs_pad = max(int(self._n_obstacles), 1)
            _d = {
                "samples": jnp.zeros((K, N, 4), jnp.float32),
                "pos0": jnp.zeros(3, jnp.float32),
                "vel0": jnp.zeros(3, jnp.float32),
                "rpy0": jnp.zeros(3, jnp.float32),
                "drpy0": jnp.zeros(3, jnp.float32),
                "T_act0": jnp.float32(self._hover_thrust),
                "ref_pos": jnp.zeros((N + 1, 3), jnp.float32),
                "ref_vel": jnp.zeros((N + 1, 3), jnp.float32),
                "hover_u": jnp.array(self._hover_u, jnp.float32),
                "prev_u": jnp.array(self._hover_u, jnp.float32),
                "inv_m": jnp.float32(1.0 / self._mass_estimate),
                "tgt_gp": jnp.zeros(3, jnp.float32),
                "tgt_gR": jnp.eye(3, dtype=jnp.float32),
                "tgt_n": jnp.array([1.0, 0.0, 0.0], jnp.float32),
                "s0": jnp.float32(0.0),
                "d0": jnp.float32(1.0),
                "tv": jnp.float32(0.0),
                "reach": jnp.float32(0.0),
                "nxt_gp": jnp.zeros(3, jnp.float32),
                "nxt_gR": jnp.eye(3, dtype=jnp.float32),
                "nv": jnp.float32(0.0),
                "n2_gp": jnp.zeros(3, jnp.float32),
                "n2_gR": jnp.eye(3, dtype=jnp.float32),
                "n2v": jnp.float32(0.0),
                "past_gp": jnp.zeros((P, 3), jnp.float32),
                "past_gR": jnp.tile(jnp.eye(3, dtype=jnp.float32), (P, 1, 1)),
                "past_mask": jnp.zeros(P, jnp.bool_),
                "obs_xy": jnp.zeros((n_obs_pad, 2), jnp.float32),
            }
            out = self._jax_eval(
                _d["samples"],
                _d["pos0"],
                _d["vel0"],
                _d["rpy0"],
                _d["drpy0"],
                _d["T_act0"],
                _d["ref_pos"],
                _d["ref_vel"],
                _d["hover_u"],
                _d["prev_u"],
                _d["inv_m"],
                _d["tgt_gp"],
                _d["tgt_gR"],
                _d["tgt_n"],
                _d["s0"],
                _d["d0"],
                _d["tv"],
                _d["reach"],
                _d["nxt_gp"],
                _d["nxt_gR"],
                _d["nv"],
                _d["n2_gp"],
                _d["n2_gR"],
                _d["n2v"],
                _d["past_gp"],
                _d["past_gR"],
                _d["past_mask"],
                _d["obs_xy"],
            )
            jax.block_until_ready(out[0])
            print(
                f"[JAX-MPPI] compiled on {jax.default_backend().upper()}. "
                f"K={K}, N={N}, n_past_pad={P}, n_obs={self._n_obstacles}."
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[JAX-MPPI] Init failed ({exc}), falling back to NumPy.")
            traceback.print_exc()
            self._jax_eval = None

    def _build_jax_fn(self) -> jax.stages.Wrapped:
        """Return jax.jit(jax.vmap(...)) that evaluates rollout + cost for all K samples."""
        # --- Fixed parameters captured in closure (won't cause recompilation) ---
        N = int(self.MPC_N)
        n_sub = int(self.ROLLOUT_SUB_STEPS)
        sub_dt = float(self.MPC_DT / n_sub)
        has_lag = bool(self._has_rotor_lag)
        lag_a = float(sub_dt / self._thrust_time_coef) if has_lag else 0.0
        has_drag = bool(self._has_drag)
        dxy = float(self._drag_xy)
        dz_coef = float(self._drag_z)
        cmd_f = float(self._cmd_f_coef)
        acc_c = float(self._acc_coef)
        g_z = float(-self._g)
        n_obs = int(self._n_obstacles)
        n_past = int(self._n_gates)

        # Cost weights
        W_IN = float(self.W_INPUT)
        W_DIN = float(self.W_DINPUT)
        W_AL = float(self.W_ATTITUDE_LIMIT)
        W_AS = float(self.W_ATTITUDE_SMOOTH)
        W_RP = float(self.W_REF_POS)
        W_RV = float(self.W_REF_VEL)
        W_ALT = float(self.W_ALTITUDE)
        W_ALTH = float(self.W_ALTITUDE_HARD)
        W_SPD = float(self.W_SPEED_LIMIT)
        W_TERM = float(self.W_TERMINAL_REF)
        W_PROG = float(self.W_PROGRESS)
        W_GDS = float(self.W_GATE_DISTANCE_STAGE)
        W_GCL_w = float(self.W_GATE_CLOSING)
        W_GAV = float(self.W_GATE_APPROACH_VEL)  # near-gate retreat penalty
        W_GALT = float(self.W_GATE_ALTITUDE)  # altitude-to-gate penalty
        W_GDT = float(self.W_GATE_DISTANCE_TERMINAL)
        W_CC = float(self.W_CROSS_CENTER)
        W_BC = float(self.W_BAD_CROSS)
        W_NC = float(self.W_NOT_CROSSED_REACHABLE)
        W_PB = float(self.W_POLE_BUFFER)
        W_PC = float(self.W_POLE_COLLISION)
        W_PN = float(self.W_POLE_NEAR_EXP)
        W_FB = float(self.W_FRAME_BUFFER)
        W_FC = float(self.W_FRAME_COLLISION)
        W_FS = float(self.W_FRAME_SLAB)
        W_FN = float(self.W_FUNNEL)
        BON = float(self.BONUS_GOOD_CROSS)
        GCM_PASS = float(self.GATE_COST_MULT_CURRENT_PASSED)
        GCM_NA = float(self.GATE_COST_MULT_NEXT_ACTIVE)
        GCM_NP = float(self.GATE_COST_MULT_NEXT_PASSIVE)
        GCM_N2 = float(self.GATE_COST_MULT_NEXT2)
        GCM_PAST = float(self.GATE_COST_MULT_PAST)
        GND = float(self.GROUND_CLEARANCE)
        CEIL = float(self.CEILING)
        V_MAX = float(self.V_MAX)
        OBS_R = float(self.OBSTACLE_RADIUS)
        OBS_B = float(self.OBSTACLE_BUFFER)
        MAX_RP = float(self.MAX_ROLL_PITCH_CMD)
        DR = float(self.DRONE_RADIUS)
        GH = float(self.GATE_HALF_OPENING)
        GS = float(self.GATE_PLANE_SLAB)
        GFB = float(self.GATE_FRAME_BUFFER)
        FL = float(self.FUNNEL_LENGTH)
        FO = float(self.FUNNEL_OUTER_HALF)
        ED = float(self.EXIT_DIST)
        EPS = float(_EPS)
        eff_half = float(max(GH - DR, 0.02))
        gate_safe = float(max(GH - DR - 0.05, 0.04))
        clear_cross = float(max(GH - float(self.GATE_CLEARANCE) - DR, 0.02))

        # Static JAX constants (shapes known at build time)
        BOX_CY = jnp.array([0.0, 0.0, -0.28, 0.28], dtype=jnp.float32)
        BOX_CZ = jnp.array([0.28, -0.28, 0.0, 0.0], dtype=jnp.float32)
        BOX_HY = jnp.array([0.36, 0.36, 0.08, 0.08], dtype=jnp.float32)
        BOX_HZ = jnp.array([0.08, 0.08, 0.36, 0.36], dtype=jnp.float32)
        BOX_HX = 0.01
        C_RPY = jnp.array(self._rpy_coef, dtype=jnp.float32)
        C_DRPY = jnp.array(self._rpy_rates_coef, dtype=jnp.float32)
        C_CMD = jnp.array(self._cmd_rpy_coef, dtype=jnp.float32)
        STAGE_W = jnp.array([0.65 + 0.70 * (k + 1) / N for k in range(N)], dtype=jnp.float32)

        def _dist_to_bars(loc3: jax.Array) -> jax.Array:
            """Distance from a single gate-local point (3,) to nearest gate bar."""
            x, y, z = loc3[0], loc3[1], loc3[2]
            dx = jnp.abs(x) - BOX_HX
            dy = jnp.abs(y - BOX_CY) - BOX_HY
            dz_b = jnp.abs(z - BOX_CZ) - BOX_HZ
            outside = jnp.sqrt(
                jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + jnp.maximum(dz_b, 0.0) ** 2
            )
            inside = jnp.maximum(dx, jnp.maximum(dy, dz_b))
            return jnp.min(jnp.where(inside < 0.0, 0.0, outside))

        def _gate_geom(
            pos3: jax.Array, gp: jax.Array, gR: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Return (c_active, c_passive, d_eff) for a single position."""
            local = (pos3 - gp) @ gR
            xa = jnp.abs(local[0])
            ya = jnp.abs(local[1])
            za = jnp.abs(local[2])
            d_bar = _dist_to_bars(local)
            d_eff = d_bar - DR
            v_soft = jnp.maximum(GFB - d_eff, 0.0)
            v_hard = jnp.maximum(-d_eff, 0.0)
            base = W_FB * (v_soft / max(GFB, EPS)) ** 2 + W_FC * (v_hard / max(DR, EPS)) ** 2
            in_sl = (xa < GS).astype(jnp.float32)
            out_op = jnp.maximum(jnp.maximum(ya - eff_half, za - eff_half), 0.0)
            sl_sc = jnp.maximum(1.0 - xa / GS, 0.0)
            base = base + W_FS * in_sl * sl_sc * (out_op / max(eff_half, EPS)) ** 2
            # active funnel
            al_a = jnp.clip(xa / max(FL, EPS), 0.0, 1.0)
            h_a = jnp.maximum(gate_safe + al_a * (FO - gate_safe), 0.02)
            c_a = base + W_FN * (1.0 - 0.55 * al_a) * (
                (jnp.maximum(ya - h_a, 0.0) / jnp.maximum(h_a, EPS)) ** 2
                + (jnp.maximum(za - h_a, 0.0) / jnp.maximum(h_a, EPS)) ** 2
            )
            # passive funnel
            al_p = jnp.clip(xa / max(ED, EPS), 0.0, 1.0)
            h_p = jnp.maximum(gate_safe + al_p * (FO - gate_safe), 0.02)
            c_p = base + 0.80 * W_FN * (1.0 - 0.50 * al_p) * (
                (jnp.maximum(ya - h_p, 0.0) / jnp.maximum(h_p, EPS)) ** 2
                + (jnp.maximum(za - h_p, 0.0) / jnp.maximum(h_p, EPS)) ** 2
            )
            return c_a, c_p, d_eff

        def _single_sample(
            u_seq: jax.Array,  # (N, 4)
            pos0: jax.Array,  # (3,)
            vel0: jax.Array,  # (3,)
            rpy0: jax.Array,  # (3,)
            drpy0: jax.Array,  # (3,)
            T_act0: jax.Array,  # scalar
            ref_pos_in: jax.Array,  # (N+1, 3)
            ref_vel_in: jax.Array,  # (N+1, 3)
            hover_u_in: jax.Array,  # (4,)
            prev_u_in: jax.Array,  # (4,)
            inv_m_in: jax.Array,  # scalar
            tgt_gp: jax.Array,  # (3,)
            tgt_gR: jax.Array,  # (3, 3)
            tgt_n: jax.Array,  # (3,)
            s0: jax.Array,  # scalar
            d0: jax.Array,  # scalar
            tv: jax.Array,  # scalar
            reach: jax.Array,  # scalar
            nxt_gp: jax.Array,  # (3,)
            nxt_gR: jax.Array,  # (3, 3)
            nv: jax.Array,  # scalar
            n2_gp: jax.Array,  # (3,)
            n2_gR: jax.Array,  # (3, 3)
            n2v: jax.Array,  # scalar
            past_gp_in: jax.Array,  # (P, 3)
            past_gR_in: jax.Array,  # (P, 3, 3)
            past_mask_in: jax.Array,  # (P,)
            obs_xy_in: jax.Array,  # (n_obs_pad, 2)
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            """Evaluate one (N,4) control sequence — vmapped over K."""

            def scan_step(carry: tuple, xs: tuple) -> tuple[tuple, jax.Array]:
                pos, vel, rpy, drpy, T_act, sp, mprog, gc, bc, lu, cost, mm = carry
                u_k, rpk, rvk, sw = xs
                pp = pos  # pos_prev

                # --- Dynamics sub-steps (Python loop → unrolled by JAX at trace time) ---
                for _ in range(n_sub):
                    if has_lag:
                        T_act = T_act + (u_k[3] - T_act) * lag_a
                        t_phys = acc_c + cmd_f * T_act
                    else:
                        t_phys = acc_c + cmd_f * u_k[3]
                    ddrpy = C_RPY * rpy + C_DRPY * drpy + C_CMD * u_k[:3]
                    phi = rpy[0]
                    theta = rpy[1]
                    psi = rpy[2]
                    cx = jnp.cos(phi)
                    sx = jnp.sin(phi)
                    cy = jnp.cos(theta)
                    sy = jnp.sin(theta)
                    cz = jnp.cos(psi)
                    sz = jnp.sin(psi)
                    R02 = sx * sz + cx * cz * sy
                    R12 = cx * sy * sz - cz * sx
                    R22 = cx * cy
                    a0 = R02 * t_phys * inv_m_in
                    a1 = R12 * t_phys * inv_m_in
                    a2 = g_z + R22 * t_phys * inv_m_in
                    if has_drag:
                        R00 = cz * cy
                        R01 = cz * sy * sx - sz * cx
                        R10 = sz * cy
                        R11 = sz * sy * sx + cz * cx
                        R20 = -sy
                        R21 = cy * sx
                        vx, vy, vz = vel[0], vel[1], vel[2]
                        vbx = R00 * vx + R10 * vy + R20 * vz
                        vby = R01 * vx + R11 * vy + R21 * vz
                        vbz = R02 * vx + R12 * vy + R22 * vz
                        fdx = dxy * vbx
                        fdy = dxy * vby
                        fdz = dz_coef * vbz
                        a0 = a0 + (R00 * fdx + R01 * fdy + R02 * fdz) * inv_m_in
                        a1 = a1 + (R10 * fdx + R11 * fdy + R12 * fdz) * inv_m_in
                        a2 = a2 + (R20 * fdx + R21 * fdy + R22 * fdz) * inv_m_in
                    drpy = drpy + ddrpy * sub_dt
                    rpy = rpy + drpy * sub_dt
                    vel = vel + jnp.array([a0, a1, a2]) * sub_dt
                    pos = pos + vel * sub_dt

                # --- Cost accumulation ---
                du = u_k - lu
                lu = u_k
                cost = cost + W_IN * jnp.sum((u_k - hover_u_in) ** 2) + W_DIN * jnp.sum(du**2)
                cost = cost + W_AL * (
                    jnp.maximum(jnp.abs(rpy[0]) - MAX_RP, 0.0) ** 2
                    + jnp.maximum(jnp.abs(rpy[1]) - MAX_RP, 0.0) ** 2
                )
                cost = cost + W_AS * jnp.sum(drpy**2)
                ep = pos - rpk
                ev = vel - rvk
                cost = cost + sw * (W_RP * jnp.sum(ep**2) + W_RV * jnp.sum(ev**2))
                bel = jnp.maximum(GND - pos[2], 0.0)
                abv = jnp.maximum(pos[2] - CEIL, 0.0)
                cost = cost + W_ALT * (bel**2 + abv**2)
                cost = cost + W_ALTH * jnp.where((pos[2] < 0.02) | (pos[2] > CEIL + 0.25), 1.0, 0.0)
                mm = jnp.minimum(mm, pos[2] - GND)
                mm = jnp.minimum(mm, CEIL - pos[2])
                cost = cost + W_SPD * jnp.maximum(jnp.linalg.norm(vel) - V_MAX, 0.0) ** 2

                # Obstacles (loop unrolled at trace time; n_obs is a Python int)
                soft_r = OBS_R + OBS_B
                for i_o in range(n_obs):
                    diff = pos[:2] - obs_xy_in[i_o]
                    d_o = jnp.sqrt(jnp.maximum(jnp.sum(diff**2), EPS))
                    cost = cost + W_PB * (jnp.maximum(soft_r - d_o, 0.0) / soft_r) ** 2
                    cost = cost + W_PC * (jnp.maximum(OBS_R - d_o, 0.0) / OBS_R) ** 2
                    cost = cost + W_PN * jnp.exp(-jnp.maximum(d_o - OBS_R, 0.0) / 0.18)
                    mm = jnp.minimum(mm, d_o - OBS_R)

                # Target gate geometry
                sn = jnp.dot(pos - tgt_gp, tgt_n)
                mprog = jnp.where(tv > 0.5, jnp.maximum(mprog, sn), mprog)
                c_ga, c_gp_, mg = _gate_geom(pos, tgt_gp, tgt_gR)
                still_a = (~gc) & (sn <= ED)
                cost = cost + jnp.where(still_a, c_ga, GCM_PASS * c_gp_) * tv
                mm = jnp.minimum(mm, jnp.where(tv > 0.5, mg, mm))
                dn = jnp.linalg.norm(pos - tgt_gp)
                dp = jnp.linalg.norm(pp - tgt_gp)
                ny = (~gc).astype(jnp.float32)
                ng_norm = jnp.maximum(d0, 1.0)
                cost = cost + ny * W_GDS * sw * (dn / ng_norm) ** 2 * tv
                cost = cost - ny * W_GCL_w * ((dp - dn) / ng_norm) * tv

                # Altitude-to-gate penalty: penalise being below target gate Z
                z_deficit = tgt_gp[2] - pos[2]  # positive = drone is below gate
                cost = cost + ny * tv * W_GALT * jnp.maximum(z_deficit - 0.05, 0.0) ** 2

                # Near-gate retreat penalty: if within 0.30 m and not yet crossed,
                # penalise negative velocity component toward gate normal.
                v_toward_gate = jnp.dot(vel, tgt_n)
                near_gate_f = (ny * tv * jnp.where(dn < 0.30, 1.0, 0.0)).astype(jnp.float32)
                cost = cost + near_gate_f * W_GAV * jnp.maximum(-v_toward_gate, 0.0)

                # Crossing detection
                cross = (~gc) & (~bc) & (sp <= 0.0) & (sn >= 0.0)
                denom = jnp.maximum(sn - sp, EPS)
                alc = jnp.clip(-sp / denom, 0.0, 1.0)
                pc = pp + alc * (pos - pp)
                lc = (pc - tgt_gp) @ tgt_gR
                yc = lc[1]
                zc = lc[2]
                err = yc * yc + zc * zc
                ins = (jnp.abs(yc) <= clear_cross) & (jnp.abs(zc) <= clear_cross)
                cf = cross.astype(jnp.float32) * tv
                cost = cost + cf * W_CC * err
                cost = cost - cf * ins.astype(jnp.float32) * BON
                ba = (
                    jnp.maximum(jnp.abs(yc) - clear_cross, 0.0) ** 2
                    + jnp.maximum(jnp.abs(zc) - clear_cross, 0.0) ** 2
                )
                cost = cost + cf * (~ins).astype(jnp.float32) * W_BC * (
                    1.0 + ba / max(clear_cross**2, EPS)
                )
                gc = jnp.where(cross & ins & (tv > 0.5), True, gc)
                bc = jnp.where(cross & (~ins) & (tv > 0.5), True, bc)
                sp = jnp.where(tv > 0.5, sn, sp)

                # Next gate (+1)
                c_na_, c_np_, mn = _gate_geom(pos, nxt_gp, nxt_gR)
                use_n = gc | (sn > ED)
                cost = cost + jnp.where(use_n, GCM_NA * c_na_, GCM_NP * c_np_) * nv
                mm = jnp.minimum(mm, jnp.where(nv > 0.5, mn, mm))

                # Gate +2
                c_n2_, _, mn2 = _gate_geom(pos, n2_gp, n2_gR)
                cost = cost + GCM_N2 * c_n2_ * n2v
                mm = jnp.minimum(mm, jnp.where(n2v > 0.5, mn2, mm))

                # Past gates (loop unrolled at trace time; n_past is a Python int)
                for i_p in range(n_past):
                    cp_, _, mpp_ = _gate_geom(pos, past_gp_in[i_p], past_gR_in[i_p])
                    pm = past_mask_in[i_p].astype(jnp.float32)
                    cost = cost + GCM_PAST * cp_ * pm
                    mm = jnp.minimum(mm, jnp.where(past_mask_in[i_p], mpp_, mm))

                new_carry = (pos, vel, rpy, drpy, T_act, sp, mprog, gc, bc, lu, cost, mm)
                return new_carry, pos  # track positions for output

            xs = (u_seq, ref_pos_in[1:], ref_vel_in[1:], STAGE_W)
            init_carry = (
                pos0.astype(jnp.float32),
                vel0.astype(jnp.float32),
                rpy0.astype(jnp.float32),
                drpy0.astype(jnp.float32),
                T_act0,
                s0,
                s0,  # sp (signed_prev), mprog
                jnp.bool_(False),  # gc  (good_crossed)
                jnp.bool_(False),  # bc  (bad_crossed)
                prev_u_in.astype(jnp.float32),  # lu  (last_u)
                jnp.float32(0.0),  # cost
                jnp.float32(jnp.inf),  # mm  (min_margin)
            )
            final, pos_seq = jax.lax.scan(scan_step, init_carry, xs)
            # pos_seq: (N, 3) — positions at steps 1..N
            all_pos = jnp.concatenate([pos0[None].astype(jnp.float32), pos_seq], axis=0)  # (N+1, 3)

            cost_f = final[10]
            mm_f = final[11]
            gc_f = final[7]
            bc_f = final[8]
            mprog_f = final[6]
            sf = final[5]

            # Terminal cost
            term = pos_seq[-1]
            cost_f = cost_f + W_TERM * jnp.sum((term - ref_pos_in[-1]) ** 2)
            cost_f = cost_f - W_PROG * (mprog_f - s0) * tv
            df = jnp.linalg.norm(term - tgt_gp)
            miss = ~gc_f
            cost_f = (
                cost_f + miss.astype(jnp.float32) * W_GDT * (df / jnp.maximum(d0, 1.0)) ** 2 * tv
            )
            cost_f = cost_f - gc_f.astype(jnp.float32) * 250.0 * tv
            cost_f = cost_f + (
                W_NC
                * miss.astype(jnp.float32)
                * jnp.maximum(jnp.float32(self.GATE_CROSS_DEPTH) - sf, 0.0) ** 2
                * tv
                * reach
            )
            mm_f = jnp.where(bc_f, jnp.minimum(mm_f, -0.10), mm_f)
            cost_f = jnp.nan_to_num(
                cost_f, nan=jnp.float32(1e12), posinf=jnp.float32(1e12), neginf=jnp.float32(-1e12)
            )
            mm_f = jnp.nan_to_num(
                mm_f, nan=jnp.float32(-1.0), posinf=jnp.float32(1e6), neginf=jnp.float32(-1.0)
            )
            return cost_f, mm_f, all_pos, gc_f, bc_f

        # Vectorise over K samples; all args except u_seq are shared across samples
        batched = jax.vmap(
            _single_sample,
            in_axes=(
                0,  # u_seq: axis 0 = K samples
                None,
                None,
                None,
                None,
                None,  # initial state: shared
                None,
                None,
                None,
                None,
                None,  # ref + hover + prev_u + inv_m
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # target gate
                None,
                None,
                None,  # next gate
                None,
                None,
                None,  # gate +2
                None,
                None,
                None,  # past gates
                None,  # obstacles
            ),
        )
        return jax.jit(batched)

    def _score_rollouts_jax(
        self,
        samples: np.ndarray,
        pos0: np.ndarray,
        vel0: np.ndarray,
        rpy0: np.ndarray,
        drpy0: np.ndarray,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """JAX-accelerated rollout+cost evaluation (dispatched when K == K_SAMPLES)."""
        K, N, _ = samples.shape
        ng = self._n_gates

        def _f32(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32)

        # Target gate
        target_valid = bool(0 <= self._target_gate < ng)
        if target_valid:
            tgt_gp = _f32(self._gate_positions[self._target_gate])
            tgt_gR = _f32(self._gate_rotmats[self._target_gate])
            tgt_n = _f32(self._get_gate_normal(self._target_gate, pos0))
            signed0 = float((pos0 - tgt_gp) @ tgt_n)
            dist0_gate = float(np.linalg.norm(pos0 - tgt_gp))
        else:
            tgt_gp = np.zeros(3, dtype=np.float32)
            tgt_gR = np.eye(3, dtype=np.float32)
            tgt_n = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            signed0 = 0.0
            dist0_gate = 1.0
        reachable = bool(
            dist0_gate
            < min(self.REACHABLE_DIST_CAP, max(0.85, 1.10 * self.V_MAX * N * self.MPC_DT))
        )

        # Next gate (+1)
        next_valid = target_valid and (self._target_gate + 1 < ng)
        nxt_gp = (
            _f32(self._gate_positions[self._target_gate + 1])
            if next_valid
            else np.zeros(3, np.float32)
        )
        nxt_gR = (
            _f32(self._gate_rotmats[self._target_gate + 1])
            if next_valid
            else np.eye(3, dtype=np.float32)
        )

        # Gate +2
        next2_valid = target_valid and (self._target_gate + 2 < ng)
        n2_gp = (
            _f32(self._gate_positions[self._target_gate + 2])
            if next2_valid
            else np.zeros(3, np.float32)
        )
        n2_gR = (
            _f32(self._gate_rotmats[self._target_gate + 2])
            if next2_valid
            else np.eye(3, dtype=np.float32)
        )

        # Past gates padded to ng
        past_gp_arr = np.zeros((ng, 3), dtype=np.float32)
        past_gR_arr = np.tile(np.eye(3, dtype=np.float32), (ng, 1, 1))
        past_mask_arr = np.zeros(ng, dtype=bool)
        valid_past = [p for p in self._past_gates if 0 <= p < ng]
        for i, pg in enumerate(valid_past[:ng]):
            past_gp_arr[i] = _f32(self._gate_positions[pg])
            past_gR_arr[i] = _f32(self._gate_rotmats[pg])
            past_mask_arr[i] = True

        # Obstacles (XY only; pad to ≥1 to avoid empty arrays)
        n_obs_pad = max(self._n_obstacles, 1)
        obs_xy = np.zeros((n_obs_pad, 2), dtype=np.float32)
        if self._n_obstacles > 0:
            obs_xy[: self._n_obstacles] = self._obstacle_positions[:, :2].astype(np.float32)

        # Call the compiled function
        out = self._jax_eval(
            jnp.array(samples, dtype=jnp.float32),
            jnp.array(pos0, dtype=jnp.float32),
            jnp.array(vel0, dtype=jnp.float32),
            jnp.array(rpy0, dtype=jnp.float32),
            jnp.array(drpy0, dtype=jnp.float32),
            jnp.float32(self._thrust_actual if self._has_rotor_lag else self._hover_thrust),
            jnp.array(ref_pos, dtype=jnp.float32),
            jnp.array(ref_vel, dtype=jnp.float32),
            jnp.array(self._hover_u, dtype=jnp.float32),
            jnp.array(self._prev_u[:4], dtype=jnp.float32),
            jnp.float32(1.0 / self._mass_estimate),
            jnp.array(tgt_gp),
            jnp.array(tgt_gR),
            jnp.array(tgt_n),
            jnp.float32(signed0),
            jnp.float32(dist0_gate),
            jnp.float32(float(target_valid)),
            jnp.float32(float(reachable)),
            jnp.array(nxt_gp),
            jnp.array(nxt_gR),
            jnp.float32(float(next_valid)),
            jnp.array(n2_gp),
            jnp.array(n2_gR),
            jnp.float32(float(next2_valid)),
            jnp.array(past_gp_arr),
            jnp.array(past_gR_arr),
            jnp.array(past_mask_arr),
            jnp.array(obs_xy),
        )
        jax.block_until_ready(out[0])

        cost = np.array(out[0], dtype=np.float64)
        min_margin = np.array(out[1], dtype=np.float64)
        all_pos = np.array(out[2])  # (K, N+1, 3) float32
        good_crossed = np.array(out[3])  # (K,) bool
        bad_crossed = np.array(out[4])  # (K,) bool

        # Add spatial corridor / progress costs in NumPy (GeometryEngine is not JAX-compatible)
        # Only evaluate on every other timestep to keep corridor overhead low for large K
        if self._geo is not None:
            for k in range(0, N, 2):
                cost += self._spatial_corridor_cost_batch(all_pos[:, k + 1, :].astype(np.float64))
            # Approximate terminal velocity from finite difference of positions
            term_vel = (all_pos[:, -1, :] - all_pos[:, -2, :]).astype(np.float64) / max(
                self.MPC_DT, _EPS
            )
            progress, curv_cost = self._spatial_progress_and_curvature(
                all_pos[:, -1, :].astype(np.float64), term_vel
            )
            cost -= self.W_SPATIAL_PROGRESS * progress
            cost += self.W_CURVATURE_SPEED * curv_cost

        return cost, min_margin, all_pos.astype(np.float32), good_crossed, bad_crossed

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
        """Core MPPI cost function: roll out all samples and compute total cost + safety margins."""
        K, N, _ = samples.shape

        # --- Dispatch to JAX path for the main K-sample MPPI batch ---
        if self._jax_eval is not None and _HAS_JAX and K == int(self.K_SAMPLES):
            return self._score_rollouts_jax(samples, pos0, vel0, rpy0, drpy0, ref_pos, ref_vel)

        # --- NumPy fallback (always used for K=1 route-ranking calls) ---
        all_pos, all_vel, all_rpy, all_drpy = self._rollout_rpy_thrust(
            samples, pos0, vel0, rpy0, drpy0
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
            reachable = dist0_gate < min(
                self.REACHABLE_DIST_CAP, max(0.85, 1.10 * self.V_MAX * N * self.MPC_DT)
            )
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

        next2_valid = target_valid and (self._target_gate + 2 < self._n_gates)
        gp_next2 = self._gate_positions[self._target_gate + 2] if next2_valid else None
        R_next2 = self._gate_rotmats[self._target_gate + 2] if next2_valid else None

        # Collect all past gate positions/rotations for frame avoidance
        past_gate_data = []
        for pg_idx in self._past_gates:
            if 0 <= pg_idx < self._n_gates:
                past_gate_data.append((self._gate_positions[pg_idx], self._gate_rotmats[pg_idx]))

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
            cost += self.W_DINPUT * np.sum(du**2, axis=1)

            # --- Attitude penalty (NEW: penalise large roll/pitch) ---
            roll_abs = np.abs(rpy_k[:, 0])
            pitch_abs = np.abs(rpy_k[:, 1])
            cost += self.W_ATTITUDE_LIMIT * np.maximum(roll_abs - self.MAX_ROLL_PITCH_CMD, 0.0) ** 2
            cost += (
                self.W_ATTITUDE_LIMIT * np.maximum(pitch_abs - self.MAX_ROLL_PITCH_CMD, 0.0) ** 2
            )
            cost += self.W_ATTITUDE_SMOOTH * np.sum(all_drpy[:, k + 1] ** 2, axis=1)

            # --- Reference tracking ---
            e_p = pos - ref_pos[k + 1]
            e_v = vel - ref_vel[k + 1]
            cost += stage_w * (
                self.W_REF_POS * np.sum(e_p**2, axis=1) + self.W_REF_VEL * np.sum(e_v**2, axis=1)
            )

            # --- Altitude safety ---
            below = np.maximum(self.GROUND_CLEARANCE - pos[:, 2], 0.0)
            above = np.maximum(pos[:, 2] - self.CEILING, 0.0)
            cost += self.W_ALTITUDE * (below**2 + above**2)
            hard_alt = (pos[:, 2] < 0.02) | (pos[:, 2] > self.CEILING + 0.25)
            cost += self.W_ALTITUDE_HARD * hard_alt
            min_margin = np.minimum(min_margin, pos[:, 2] - self.GROUND_CLEARANCE)
            min_margin = np.minimum(min_margin, self.CEILING - pos[:, 2])

            # --- Speed limit ---
            speed = np.linalg.norm(vel, axis=1)
            overspeed = np.maximum(speed - self.V_MAX, 0.0)
            cost += self.W_SPEED_LIMIT * overspeed**2

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
                d = np.sqrt(np.maximum(np.sum(diff_xy**2, axis=2), _EPS))  # (K, n_obs)
                soft_r = self.OBSTACLE_RADIUS + self.OBSTACLE_BUFFER
                v_soft = np.maximum(soft_r - d, 0.0)
                v_hard = np.maximum(self.OBSTACLE_RADIUS - d, 0.0)
                cost += self.W_POLE_BUFFER * np.sum((v_soft / soft_r) ** 2, axis=1)
                cost += self.W_POLE_COLLISION * np.sum((v_hard / self.OBSTACLE_RADIUS) ** 2, axis=1)
                cost += self.W_POLE_NEAR_EXP * np.sum(
                    np.exp(-np.maximum(d - self.OBSTACLE_RADIUS, 0.0) / 0.18), axis=1
                )
                min_margin = np.minimum(min_margin, np.min(d - self.OBSTACLE_RADIUS, axis=1))

            # --- Gate geometry ---
            if target_valid:
                signed_now = (pos - gp) @ normal
                max_progress = np.maximum(max_progress, signed_now)

                still_approaching = ~(good_crossed | (signed_now > self.EXIT_DIST))
                if np.any(still_approaching):
                    c_gate, m_gate = self._gate_geometry_cost_and_margin(
                        pos[still_approaching], gp, R, active_funnel=True
                    )
                    cost[still_approaching] += c_gate
                    min_margin[still_approaching] = np.minimum(
                        min_margin[still_approaching], m_gate
                    )
                if np.any(~still_approaching):
                    c_gate_p, m_gate_p = self._gate_geometry_cost_and_margin(
                        pos[~still_approaching], gp, R, active_funnel=False
                    )
                    cost[~still_approaching] += self.GATE_COST_MULT_CURRENT_PASSED * c_gate_p
                    min_margin[~still_approaching] = np.minimum(
                        min_margin[~still_approaching], m_gate_p
                    )

                dist_gate_now = np.linalg.norm(pos - gp, axis=1)
                dist_gate_prev = np.linalg.norm(pos_prev - gp, axis=1)
                not_crossed_yet = ~good_crossed
                norm_gate = max(dist0_gate, 1.0)
                cost += (
                    not_crossed_yet
                    * self.W_GATE_DISTANCE_STAGE
                    * stage_w
                    * (dist_gate_now / norm_gate) ** 2
                )
                cost -= (
                    not_crossed_yet
                    * self.W_GATE_CLOSING
                    * ((dist_gate_prev - dist_gate_now) / norm_gate)
                )

                # Altitude-to-gate penalty: penalise being below target gate Z
                z_deficit = gp[2] - pos[:, 2]  # positive = drone is below gate
                cost += (
                    not_crossed_yet * self.W_GATE_ALTITUDE * np.maximum(z_deficit - 0.05, 0.0) ** 2
                )

                # Near-gate velocity penalty: penalise retreating (v_toward < 0) when
                # within 0.30 m and not yet crossed.  Uses the velocity at this step.
                if target_valid and self.W_GATE_APPROACH_VEL > 0.0:
                    step_vel = (pos - pos_prev) / max(
                        self.MPC_DT / max(self.ROLLOUT_SUB_STEPS, 1), _EPS
                    )
                    v_toward = step_vel @ normal  # scalar — same normal for all K
                    near_gate = not_crossed_yet & (dist_gate_now < 0.30)
                    cost += near_gate * self.W_GATE_APPROACH_VEL * np.maximum(-v_toward, 0.0)

                crossing = (
                    (~good_crossed) & (~bad_crossed) & (signed_prev <= 0.0) & (signed_now >= 0.0)
                )
                if np.any(crossing):
                    denom = np.maximum(signed_now - signed_prev, _EPS)
                    alpha = np.clip(-signed_prev / denom, 0.0, 1.0)
                    p_cross = pos_prev + alpha[:, None] * (pos - pos_prev)
                    local_cross = self._world_to_gate_local(p_cross, gp, R)
                    y_c = local_cross[:, 1]
                    z_c = local_cross[:, 2]
                    err = y_c * y_c + z_c * z_c
                    clear = max(
                        self.GATE_HALF_OPENING - self.GATE_CLEARANCE - self.DRONE_RADIUS, 0.02
                    )
                    inside = (np.abs(y_c) <= clear) & (np.abs(z_c) <= clear)

                    cost += crossing * self.W_CROSS_CENTER * err
                    cost += crossing * inside * (-self.BONUS_GOOD_CROSS)
                    bad_amount = (
                        np.maximum(np.abs(y_c) - clear, 0.0) ** 2
                        + np.maximum(np.abs(z_c) - clear, 0.0) ** 2
                    )
                    cost += (
                        crossing
                        * (~inside)
                        * (self.W_BAD_CROSS * (1.0 + bad_amount / (clear * clear)))
                    )
                    good_crossed |= crossing & inside
                    bad_crossed |= crossing & (~inside)

                signed_prev = signed_now

            if next_valid and gp_next is not None and R_next is not None:
                use_next = (
                    good_crossed | (signed_now > self.EXIT_DIST)
                    if target_valid
                    else np.zeros(K, dtype=bool)
                )
                if np.any(use_next):
                    cn, mn = self._gate_geometry_cost_and_margin(
                        pos[use_next], gp_next, R_next, True
                    )
                    cost[use_next] += self.GATE_COST_MULT_NEXT_ACTIVE * cn
                    min_margin[use_next] = np.minimum(min_margin[use_next], mn)
                if np.any(~use_next):
                    cn2, mn2 = self._gate_geometry_cost_and_margin(
                        pos[~use_next], gp_next, R_next, False
                    )
                    cost[~use_next] += self.GATE_COST_MULT_NEXT_PASSIVE * cn2
                    min_margin[~use_next] = np.minimum(min_margin[~use_next], mn2)

            # --- Gate +2 frame avoidance ---
            if next2_valid and gp_next2 is not None and R_next2 is not None:
                cn2f, mn2f = self._gate_geometry_cost_and_margin(pos, gp_next2, R_next2, False)
                cost += self.GATE_COST_MULT_NEXT2 * cn2f
                min_margin = np.minimum(min_margin, mn2f)

            # --- All past gate frames ---
            # On U-turns and dips the drone can loop back near any
            # previously passed gate frame. Check all of them. The cost
            # function is distance-based so far-away gates contribute ~0.
            for gp_past, R_past in past_gate_data:
                cp, mp = self._gate_geometry_cost_and_margin(
                    pos, gp_past, R_past, active_funnel=False
                )
                cost += self.GATE_COST_MULT_PAST * cp
                min_margin = np.minimum(min_margin, mp)

        # --- Terminal ---
        terminal_pos = all_pos[:, -1]
        terminal_error = terminal_pos - ref_pos[-1]
        cost += self.W_TERMINAL_REF * np.sum(terminal_error**2, axis=1)

        if target_valid:
            signed_final = (terminal_pos - gp) @ normal
            cost -= self.W_PROGRESS * (max_progress - signed0)
            dist_final = np.linalg.norm(terminal_pos - gp, axis=1)
            miss = ~good_crossed
            cost += miss * self.W_GATE_DISTANCE_TERMINAL * (dist_final / max(dist0_gate, 1.0)) ** 2
            cost -= good_crossed * 250.0

            if reachable:
                cost += (
                    self.W_NOT_CROSSED_REACHABLE
                    * miss
                    * np.maximum(self.GATE_CROSS_DEPTH - signed_final, 0.0) ** 2
                )

        min_margin = np.where(bad_crossed, np.minimum(min_margin, -0.10), min_margin)

        cost = np.nan_to_num(cost, nan=1e12, posinf=1e12, neginf=-1e12)
        min_margin = np.nan_to_num(min_margin, nan=-1.0, posinf=1e6, neginf=-1.0)

        return cost, min_margin, all_pos.astype(np.float32), good_crossed, bad_crossed

    # ------------------------------------------------------------------
    #  Sampling in 4-D control space
    # ------------------------------------------------------------------
    def _sample_sequences(self, route_seqs: list[np.ndarray]) -> np.ndarray:
        """Generate K MPPI samples around route-anchored centres plus deterministic variants."""
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

        center_bank = cleaned[: max(1, min(len(cleaned), self.ROUTE_TOP_K))]
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
        up[:, 3] += 0.015  # meaningful climb bias
        down[:, 3] -= 0.010  # descent bias
        deterministic.extend([up, down])

        # Climb-priority variant: reduce forward pitch, add thrust
        climb_variant = cleaned[0].copy()
        climb_variant[:, 1] *= 0.7  # reduce pitch → less forward speed, more vertical budget
        climb_variant[:, 3] += 0.012  # add thrust for climb
        deterministic.append(climb_variant)

        write = n_rand
        det_i = 0
        while write < K:
            samples[write] = deterministic[det_i % len(deterministic)]
            write += 1
            det_i += 1

        return self._clip_u(samples)

    def _shift_distribution(self):
        """Shift MPPI mean and sigma one step forward in time (warm-start between replans)."""
        self._mean_u[:-1] = self._mean_u[1:]
        self._mean_u[-1] = self._mean_u[-2]
        self._mean_u = self._clip_u(self._mean_u)
        self._sigma[:-1] = self._sigma[1:]
        self._sigma[-1] = np.maximum(self._sigma[-2], self.SIGMA_INIT)
        self._sigma = np.clip(self._sigma, self.SIGMA_MIN, self.SIGMA_MAX)

    def _update_distribution(
        self, samples: np.ndarray, cost: np.ndarray, min_margin: np.ndarray
    ) -> tuple[np.ndarray, int, bool]:
        """Update MPPI mean and sigma from weighted sample costs (distribution fitting)."""
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
        elite_std = np.sqrt(np.mean(elite_delta**2, axis=0) + 1e-8)
        target_sigma = np.clip(1.35 * elite_std, self.SIGMA_MIN, self.SIGMA_MAX)
        self._sigma = np.clip(
            0.82 * self._sigma + 0.18 * target_sigma, self.SIGMA_MIN, self.SIGMA_MAX
        )
        self._mean_u = new_mean

        return new_mean, best_idx, bool(np.any(safe_mask))

    # ------------------------------------------------------------------
    #  Candidate selection (from scenario MPC)
    # ------------------------------------------------------------------
    def _crossing_yz_from_trajectories(
        self, trajectories: np.ndarray, pos0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gate-local YZ radius at the crossing point for each trajectory."""
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

    def _candidate_progress_cost(
        self,
        indices: np.ndarray,
        cost: np.ndarray,
        trajectories: np.ndarray,
        pos: np.ndarray,
        min_margin: np.ndarray,
        cross_yz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Augmented selection cost: base + margin penalty + crossing quality + progress."""
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

    def _choose_fresh_robust_candidates(
        self,
        cost: np.ndarray,
        min_margin: np.ndarray,
        trajectories: np.ndarray,
        good_crossed: np.ndarray,
        bad_crossed: np.ndarray,
        pos: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Pick candidates hierarchically: robust center-crossings first, then safe/emergency."""
        cross_yz, geometric_crossed = self._crossing_yz_from_trajectories(trajectories, pos)
        noncrash = (~bad_crossed) & (min_margin > self.SAFE_SELECTION_MARGIN)
        robust = (~bad_crossed) & (min_margin > self.SELECTION_MARGIN)

        for label, mask in [
            (
                "robust_center_cross",
                robust & good_crossed & geometric_crossed & (cross_yz <= self.ROBUST_CROSS_RADIUS),
            ),
            (
                "robust_cross",
                robust & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS),
            ),
            (
                "loose_center_cross",
                noncrash
                & good_crossed
                & geometric_crossed
                & (cross_yz <= self.ROBUST_CROSS_RADIUS),
            ),
            (
                "loose_cross",
                noncrash & good_crossed & geometric_crossed & (cross_yz <= self.LOOSE_CROSS_RADIUS),
            ),
            ("robust_progress", robust),
            ("loose_progress", noncrash),
        ]:
            if np.any(mask):
                pool = np.where(mask)[0]
                sel_cost = self._candidate_progress_cost(
                    pool, cost, trajectories, pos, min_margin, cross_yz
                )
                order = pool[np.argsort(sel_cost)]
                keep = max(1, min(int(self.ELITE_BLEND_COUNT), len(order)))
                return order[:keep].astype(int), label

        pool = np.arange(cost.shape[0])
        sel_cost = self._candidate_progress_cost(
            pool, cost, trajectories, pos, min_margin, cross_yz
        )
        order = pool[np.lexsort((sel_cost, -min_margin[pool]))]
        return order[:1].astype(int), "emergency"

    def _elite_weighted_sequence(
        self, samples: np.ndarray, cost: np.ndarray, candidate_idx: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        """Blend top-K elite candidates into a single control sequence (softmax-weighted)."""
        if candidate_idx.size == 0:
            best_idx = int(np.argmin(cost))
            return self._clip_u(samples[best_idx].copy()), best_idx, 1
        order = candidate_idx[np.argsort(cost[candidate_idx])]
        elite = order[: max(1, min(int(self.ELITE_BLEND_COUNT), len(order)))]
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
    def _rank_routes_by_rollout(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        drpy: np.ndarray,
        routes: list[np.ndarray],
        route_costs: list[float],
    ) -> tuple[list[np.ndarray], list[float], list[np.ndarray], np.ndarray, np.ndarray]:
        """Rank candidate routes by rolling out PD sequences and scoring with full dynamics."""
        if not routes:
            ref_pos, ref_vel = self._generate_references(pos, vel, None)
            seq = self._make_pd_sequence(pos, vel, rpy, drpy, ref_pos, ref_vel)
            return [], [], [seq], ref_pos, ref_vel

        records = []
        for idx, route in enumerate(routes):
            r_pos, r_vel = self._generate_references(pos, vel, route)
            seq = self._make_pd_sequence(pos, vel, rpy, drpy, r_pos, r_vel)
            c, margin, traj, crossed, bad = self._score_rollouts(
                seq[None, :, :], pos, vel, rpy, drpy, r_pos, r_vel
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
            records.append(
                (
                    bucket,
                    select_score,
                    idx,
                    route,
                    geom,
                    seq,
                    r_pos,
                    r_vel,
                    c0,
                    m0,
                    crossed0,
                    bad0,
                    traj[0].astype(np.float64),
                )
            )

        records.sort(key=lambda item: (item[0], item[1]))
        keep = records[: max(1, int(self.ROUTE_TOP_K))]
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
        """APF reactive obstacle avoidance: translate repulsive forces into attitude adjustments."""
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
            out[0] = float(
                np.clip(out[0] + d_roll, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
            )
            out[1] = float(
                np.clip(out[1] + d_pitch, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD)
            )
            return out
        return u0.copy()

    def _yaw_command(self, obs: dict[str, "NDArray[np.floating]"], pos: np.ndarray) -> float:
        """Compute yaw command that smoothly tracks the direction toward the target gate."""
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

    def _finish_action(
        self, obs: dict[str, "NDArray[np.floating]"], pos: np.ndarray, u0: np.ndarray
    ) -> np.ndarray:
        """Clip, apply altitude safety guards, and return the [roll, pitch, yaw, thrust] action."""
        roll_cmd = float(np.clip(u0[0], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        pitch_cmd = float(np.clip(u0[1], -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        yaw_cmd = float(np.clip(u0[2], -self.MAX_YAW_CMD, self.MAX_YAW_CMD))
        thrust_cmd = float(np.clip(u0[3], self._thrust_min, self._thrust_max))

        # Altitude safety guards
        if pos[2] < self.GROUND_CLEARANCE + 0.08:
            thrust_cmd += (
                0.65 * (self.GROUND_CLEARANCE + 0.08 - pos[2]) * self._mass_estimate * self._g
            )
        if pos[2] > self.CEILING - 0.18:
            overshoot = pos[2] - (self.CEILING - 0.18)
            thrust_cmd -= (0.55 + 2.2 * overshoot) * overshoot * self._mass_estimate * self._g
        thrust_cmd = float(np.clip(thrust_cmd, self._thrust_min, self._thrust_max))

        self._prev_output = np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd], dtype=np.float64)
        return self._prev_output.astype(np.float32)

    # ------------------------------------------------------------------
    #  Route tracking feedback (translated to attitude space)
    # ------------------------------------------------------------------
    def _route_tracking_feedback(
        self,
        u_base: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        elapsed: float,
        ref_pos: np.ndarray | None = None,
        ref_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Closed-loop PD correction tracking the selected route between replans."""
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
            p_ref, tangent, seg_idx, _ = self._project_polyline_tracking_target(
                path, pos, lookahead
            )
            e_p = p_ref - pos
            e_v = np.zeros(3, dtype=np.float64)
            # Desired along-track velocity
            desired_speed = self.V_CRUISE
            if 0 <= self._target_gate < self._n_gates:
                d_gate = float(np.linalg.norm(pos - self._gate_positions[self._target_gate]))
                if d_gate < self.ALIGN_START_DIST:
                    blend = 1.0 - d_gate / max(self.ALIGN_START_DIST, _EPS)
                    desired_speed = min(
                        desired_speed, self.V_CRUISE - blend * (self.V_CRUISE - self.V_GATE)
                    )
            e_v = tangent * desired_speed - vel

        # Convert position/velocity error to desired acceleration correction
        kp_xy, kp_z = 3.2, 3.5
        kd_xy, kd_z = 2.0, 2.2
        a_corr = np.array(
            [
                kp_xy * e_p[0] + kd_xy * e_v[0],
                kp_xy * e_p[1] + kd_xy * e_v[1],
                kp_z * e_p[2] + kd_z * e_v[2],
            ]
        )

        # Convert acceleration correction to attitude perturbation (small-angle)
        # roll ~ -ay / g, pitch ~ ax / g
        d_roll = float(np.clip(-a_corr[1] / self._g, -0.15, 0.15))
        d_pitch = float(np.clip(a_corr[0] / self._g, -0.15, 0.15))
        d_thrust = float(np.clip(a_corr[2] * self._mass_estimate, -0.08, 0.08))

        out = np.asarray(u_base, dtype=np.float64).copy()
        out[0] = float(np.clip(out[0] + d_roll, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        out[1] = float(np.clip(out[1] + d_pitch, -self.MAX_ROLL_PITCH_CMD, self.MAX_ROLL_PITCH_CMD))
        out[3] = float(np.clip(out[3] + d_thrust, self._thrust_min, self._thrust_max))
        return out

    def _project_polyline_tracking_target(
        self, polyline: np.ndarray, pos: np.ndarray, lookahead_dist: float
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Find the closest point on a polyline and return a lookahead target for path tracking."""
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return pos.copy(), np.array([1.0, 0.0, 0.0]), 0, 0.0

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        best_d2, best_i, best_s = math.inf, 0, 0.0
        for i in range(len(seg)):
            if seg_len[i] <= 1e-6:
                continue
            v = seg[i]
            t = float(np.clip(np.dot(pos - pts[i], v) / max(float(np.dot(v, v)), _EPS), 0.0, 1.0))
            q = pts[i] + t * v
            d2 = float(np.dot(pos - q, pos - q))
            if d2 < best_d2:
                best_d2, best_i = d2, i
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
    def _check_compute_lag(self, t_start: float) -> None:
        """Warn when compute_control exceeds the real-time control period."""
        elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
        self._compute_total_count += 1
        if self._compute_total_count == 1:
            self._compute_ms_ema = elapsed_ms
        else:
            self._compute_ms_ema = 0.90 * self._compute_ms_ema + 0.10 * elapsed_ms
        deadline_ms = 1000.0 * self._compute_deadline_s
        if elapsed_ms > deadline_ms:
            self._compute_lag_count += 1
            lag_rate = 100.0 * self._compute_lag_count / max(1, self._compute_total_count)
            print(
                f"[SPATIAL-LAG] WARNING controller lagging: "
                f"compute={elapsed_ms:.1f}ms > deadline={deadline_ms:.1f}ms "
                f"(tick={self._tick}, lag "
                f"{self._compute_lag_count}/{self._compute_total_count}={lag_rate:.1f}%, "
                f"ema={self._compute_ms_ema:.1f}ms)"
            )

    def compute_control(
        self, obs: dict[str, "NDArray[np.floating]"], info: dict | None = None
    ) -> "NDArray[np.floating]":
        """Replan with MPPI or interpolate the cached plan, then output attitude+thrust."""
        _compute_t_start = time.perf_counter()
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
            t_phys_prev = self._acc_coef + self._cmd_f_coef * self._prev_u[3]
            R22_est = math.cos(rpy[0]) * math.cos(rpy[1])
            if R22_est > 0.5 and t_phys_prev > 0.05:
                m_est = t_phys_prev * R22_est / self._g
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
                self._last_planner_tick = -(10**9)
                self._cached_plan_tick = -(10**9)
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
            action = self._finish_action(obs, pos, u0)
            self._check_compute_lag(_compute_t_start)
            return action

        # --- Expensive replan ---
        plan_t0 = time.perf_counter()
        previous_plan_tick = self._cached_plan_tick
        self._last_planner_tick = self._tick

        if not self.FORGET_OLD_SCENARIOS and previous_plan_tick > -(10**8):
            elapsed = max(0.0, (self._tick - previous_plan_tick) * self._dt)
            shift_steps = min(int(elapsed / max(self.MPC_DT, _EPS)), self.MPC_N - 1)
            for _ in range(shift_steps):
                self._shift_distribution()

        # Build and rank route candidates
        raw_routes, raw_route_costs = self._build_route_candidates(pos, vel)
        routes, route_costs, route_seqs, ref_pos, ref_vel = self._rank_routes_by_rollout(
            pos, vel, rpy, drpy, raw_routes, raw_route_costs
        )

        self._route_candidates = routes
        self._route_costs = route_costs
        self._active_route_points = routes[0].copy() if routes else None

        if route_seqs:
            if self.FORGET_OLD_SCENARIOS:
                self._mean_u = self._clip_u(route_seqs[0].copy())
            else:
                self._mean_u = self._clip_u(
                    (1.0 - self.ROUTE_MEAN_BLEND) * self._mean_u
                    + self.ROUTE_MEAN_BLEND * route_seqs[0]
                )

        # Inject the warped (discovery-shifted) cached sequence as a candidate
        # so MPPI can choose between the fresh plan and the smoothly adapted old plan.
        if hasattr(self, "_warped_discovery_seq") and self._warped_discovery_seq is not None:
            route_seqs.append(self._warped_discovery_seq)
            self._warped_discovery_seq = None

        # Sample and score with full RPY dynamics
        samples = self._sample_sequences(route_seqs)
        cost, min_margin, trajectories, good_crossed, bad_crossed = self._score_rollouts(
            samples, pos, vel, rpy, drpy, ref_pos, ref_vel
        )
        new_mean, best_idx, any_safe = self._update_distribution(samples, cost, min_margin)

        candidate_idx, selection_label = self._choose_fresh_robust_candidates(
            cost, min_margin, trajectories, good_crossed, bad_crossed, pos
        )
        selected_sequence, selected_idx, elite_count = self._elite_weighted_sequence(
            samples, cost, candidate_idx
        )

        chosen_u0 = selected_sequence[0].copy()
        route_u0 = route_seqs[0][0].copy() if route_seqs else chosen_u0.copy()
        best_margin = float(min_margin[selected_idx])
        robust_selected = best_margin > self.SELECTION_MARGIN and not bool(
            bad_crossed[selected_idx]
        )

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
            selected_sequence.copy()
            if self.FORGET_OLD_SCENARIOS
            else 0.70 * selected_sequence + 0.30 * new_mean
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
            anchor_ok = (
                "?" if anchor_m is None else ("OK" if anchor_m > 0.01 else f"BAD({anchor_m:.3f})")
            )
            n_safe = int(np.sum(min_margin > 0.0)) if min_margin is not None else 0
            gates_passed = self._n_gates if self._target_gate < 0 else int(self._target_gate)
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

        # --- Detailed gate-approach diagnostics (every replan tick when close to gate) ---
        if self.DEBUG_PRINT_ENABLED and 0 <= self._target_gate < self._n_gates:
            _gp_d = self._gate_positions[self._target_gate]
            _n_d = self._get_gate_normal(self._target_gate, pos)
            _dg_d = float(np.linalg.norm(pos - _gp_d))
            if _dg_d < 0.70:
                _term = trajectories[:, -1, :].astype(np.float64)
                _sf = (_term - _gp_d) @ _n_d  # signed distance past gate (+ = past)
                _good_pct = 100.0 * float(np.mean(good_crossed))
                _bad_pct = 100.0 * float(np.mean(bad_crossed))
                _vt = float(vel @ _n_d)  # current velocity toward gate
                _reach = _dg_d < min(
                    self.REACHABLE_DIST_CAP, max(0.85, 1.10 * self.V_MAX * self.MPC_N * self.MPC_DT)
                )
                _sel_sf = float((_term[int(selected_idx)] - _gp_d) @ _n_d)
                _sf_pos_pct = 100.0 * float(np.mean(_sf > 0.0))  # % rollouts reaching past gate
                print(
                    f"[GATE-DIAG] step={self._tick:04d} gate={self._target_gate} "
                    f"dg={_dg_d:.3f}m v_toward={_vt:+.3f}m/s reachable={_reach} "
                    f"good={_good_pct:.1f}% bad={_bad_pct:.1f}% past_gate={_sf_pos_pct:.1f}% "
                    f"sf_min={_sf.min():+.3f} sf_mean={_sf.mean():+.3f} sf_max={_sf.max():+.3f} "
                    f"sel_sf={_sel_sf:+.3f} plan={self._last_plan_ms:.1f}ms"
                )

        action = self._finish_action(obs, pos, u0)
        self._check_compute_lag(_compute_t_start)
        return action

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _print_episode_summary(self, new_target: int, terminated: bool, truncated: bool) -> None:
        """Print one-line episode outcome (FINISHED/CRASH/TIMEOUT) regardless of debug flag."""
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
        lag_rate = 100.0 * self._compute_lag_count / max(1, self._compute_total_count)
        print(
            f"[SPATIAL-EP] {outcome:<8s} gates={gates_passed}/{self._n_gates} "
            f"time={flight_time:5.2f}s ticks={self._tick:4d} "
            f"plans={self._plan_counter:4d} plan_ms_ema={self._plan_ms_ema:5.1f} "
            f"compute_ms_ema={self._compute_ms_ema:5.1f} "
            f"lag={self._compute_lag_count}/{self._compute_total_count} ({lag_rate:.1f}%)"
        )

    def step_callback(
        self,
        action: "NDArray[np.floating]",
        obs: dict[str, "NDArray[np.floating]"],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Post-step: update gate targets, discover gate/obstacle positions, trigger replan."""
        new_target = int(obs["target_gate"])
        gate_changed = new_target != self._target_gate

        # A gate was just passed: add it to the past-gates list so the
        # MPPI cost keeps penalising ALL passed frames (U-turns/dips).
        if (
            gate_changed
            and 0 <= self._target_gate < self._n_gates
            and new_target > self._target_gate
        ):
            if self._target_gate not in self._past_gates:
                self._past_gates.append(self._target_gate)

        self._target_gate = new_target

        if gate_changed:
            self._sigma = np.maximum(self._sigma, self.SIGMA_INIT[None, :])
            self._last_planner_tick = -(10**9)
            self._cached_plan_tick = -(10**9)
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
            warped_seq = self._warp_cached_sequence_for_discovery(gate_deltas, obs_deltas)
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
            self._last_planner_tick = -(10**9)
            self._cached_plan_tick = -(10**9)
            self._cached_u_index = 0

        return self._finished

    # ------------------------------------------------------------------
    #  Trajectory warping on object discovery
    # ------------------------------------------------------------------
    def _warp_cached_sequence_for_discovery(
        self, gate_deltas: list[tuple[int, np.ndarray]], obs_deltas: list[tuple[int, np.ndarray]]
    ) -> np.ndarray | None:
        """Shift the cached control sequence toward newly discovered gate/obstacle positions."""
        if self._cached_u_sequence is None or len(gate_deltas) == 0 and len(obs_deltas) == 0:
            return None

        # We need a predicted trajectory from the old mean sequence
        old_seq = self._cached_u_sequence.copy()
        pos0 = self._last_pos.copy()
        vel0 = np.zeros(3, dtype=np.float64)  # approximate
        rpy0 = self._last_rpy.copy()
        drpy0 = self._last_drpy.copy()

        # Quick single-sample rollout of the old sequence
        old_traj, _, _, _ = self._rollout_rpy_thrust(old_seq[None, :, :], pos0, vel0, rpy0, drpy0)
        old_traj = old_traj[0]  # (N+1, 3)

        N = self.MPC_N
        # Accumulate desired position corrections at each horizon step
        pos_correction = np.zeros((N + 1, 3), dtype=np.float64)

        for gi, delta in gate_deltas:
            gate_pos = self._gate_positions[gi]  # already updated
            for k in range(N + 1):
                dist = float(np.linalg.norm(old_traj[k] - (gate_pos - delta)))
                w = math.exp(-0.5 * (dist / self.WARP_GATE_INFLUENCE_RADIUS) ** 2)
                pos_correction[k] += w * delta

        for oi, delta in obs_deltas:
            obs_pos = self._obstacle_positions[oi]  # already updated
            for k in range(N + 1):
                dist = float(np.linalg.norm(old_traj[k] - (obs_pos - delta)))
                if dist < _EPS:
                    continue
                # Push away from obstacle's new position
                w = math.exp(-0.5 * (dist / self.WARP_OBS_INFLUENCE_RADIUS) ** 2)
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
            d_thrust = float(
                np.clip(a_corr[2] * self._mass_estimate / self._cmd_f_coef, -0.02, 0.02)
            )

            warped[k, 0] += d_roll
            warped[k, 1] += d_pitch
            warped[k, 3] += d_thrust

        return self._clip_u(warped)

    def render_callback(self, sim: "Sim"):
        """Visualise active route, best trajectory, and gate targets in the simulator."""
        from crazyflow.sim.visualize import draw_line, draw_points

        if self._active_route_points is not None and len(self._active_route_points) > 1:
            draw_line(
                sim,
                self._active_route_points,
                rgba=(1.0, 0.85, 0.1, 0.90),
                start_size=2.2,
                end_size=2.2,
            )

        if self._last_best_traj is not None and len(self._last_best_traj) > 1:
            draw_line(
                sim, self._last_best_traj, rgba=(0.0, 1.0, 0.2, 0.85), start_size=2.0, end_size=2.0
            )

        if self._last_route_anchor_traj is not None and len(self._last_route_anchor_traj) > 1:
            draw_line(
                sim,
                self._last_route_anchor_traj,
                rgba=(0.0, 0.9, 1.0, 0.50),
                start_size=1.5,
                end_size=1.5,
            )

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            draw_points(sim, gp.reshape(1, -1), rgba=(1.0, 1.0, 0.0, 1.0), size=0.04)
