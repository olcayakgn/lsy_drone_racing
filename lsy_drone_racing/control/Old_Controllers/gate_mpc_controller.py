"""Gate-target MPC drone racing controller.

v5 changes compared with the previous version:
  - Fixed gate-normal flipping near the gate plane by caching one planning frame
    per gate. The current gate no longer changes approach direction when the
    drone is close to or slightly past the plane.
  - Removed the nonconvex near-hard gate-frame bar constraints from the active
    OCP constraints. The gate is protected by the local-y/local-z funnel/opening
    constraints instead. This avoids the SQP-RTI linearization wall that can make
    the drone hover before gate 0.
  - Added reference-trajectory warm starting. SQP-RTI now linearizes around the
    intended route instead of around a stale hovering solution.
  - Added a small gate-progress acceleration when the drone is already in the
    gate corridor. This prevents hesitation directly before the gate.
  - Kept obstacle constraints soft but expensive, with less over-inflation so the
    first gate remains feasible.

Architecture:
  The MPC IS the planner. No trajectory planner needed.
  - Reference: route through current gate and, when visible in horizon, next gate
  - Velocity reference: tangent to the chosen route, not simply toward gate center
  - MPC optimizes the path considering drone dynamics + obstacle/gate constraints

Important modeling convention:
  State: [pos(3), vel(3), u_prev(3)] = 9
  Control: [ax, ay, az] = 3

  Here ax, ay, az are WORLD-FRAME specific thrust components.
  Therefore:
      vx_dot = ax
      vy_dot = ay
      vz_dot = az - g

  Hover is:
      u = [0, 0, 9.81]

Requires:
  - acados
  - MinGW-w64 in PATH on Windows
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rot

# ---- acados / casadi setup ----
os.environ.setdefault("ACADOS_SOURCE_DIR", r"C:\Users\Q678730\Downloads\acados-main\acados-main")
os.environ.setdefault("ACADOS_INSTALL_DIR", r"C:\Users\Q678730\Downloads\acados-main\acados-main")
_acados_bin = r"C:\Users\Q678730\Downloads\acados-main\acados-main\bin"
_acados_lib = r"C:\Users\Q678730\Downloads\acados-main\acados-main\lib"
_mingw_bin = r"C:\mingw64\bin"

for _p in [_mingw_bin, _acados_lib, _acados_bin]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_acados_bin)
    os.add_dll_directory(_acados_lib)
    os.add_dll_directory(_mingw_bin)

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


_EPS = 1e-10
_N_OBS_SLOTS = 4
_N_GATE_POST_SLOTS = 4
_TOTAL_CYL = _N_OBS_SLOTS + _N_GATE_POST_SLOTS
_NH = _TOTAL_CYL + 2
_N_PARAMS = _TOTAL_CYL * 2 + 3 + 9

_GATE_FRAME_OUTER_HALF_MPC = 0.28
_CONSTRAINT_OFF = 1.0e6


def _create_mpc_model() -> AcadosModel:
    """Double-integrator with parametric obstacle/gate constraints.

    Constraint order:
        0 .. _N_OBS_SLOTS-1:
            vertical pole / obstacle cylinders in world XY

        _N_OBS_SLOTS .. _TOTAL_CYL-1:
            gate-frame bar expressions. In v5 these are normally disabled in
            the OCP bounds because their nonconvex SQP-RTI linearization can
            form an artificial wall in front of the active gate.

        _TOTAL_CYL:
            gate-local y opening/funnel coordinate

        _TOTAL_CYL + 1:
            gate-local z opening/funnel coordinate
    """
    px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
    vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
    ax_prev, ay_prev, az_prev = cs.MX.sym("ax_prev"), cs.MX.sym("ay_prev"), cs.MX.sym("az_prev")

    x = cs.vertcat(px, py, pz, vx, vy, vz, ax_prev, ay_prev, az_prev)

    ax, ay, az = cs.MX.sym("ax"), cs.MX.sym("ay"), cs.MX.sym("az")
    u = cs.vertcat(ax, ay, az)

    p = cs.MX.sym("p", _N_PARAMS)

    g = 9.81
    tau = 0.05

    x_dot = cs.vertcat(
        vx,
        vy,
        vz,
        ax,
        ay,
        az - g,
        (ax - ax_prev) / tau,
        (ay - ay_prev) / tau,
        (az - az_prev) / tau,
    )

    h_list = []

    # World-XY vertical pole constraints.
    for i in range(_N_OBS_SLOTS):
        ox, oy = p[2 * i], p[2 * i + 1]
        h_list.append((px - ox) ** 2 + (py - oy) ** 2)

    gi = _TOTAL_CYL * 2
    gx, gy, gz = p[gi], p[gi + 1], p[gi + 2]

    ri = gi + 3
    dx, dy, dz = px - gx, py - gy, pz - gz

    # R is stored row-major. local = R.T @ (p_world - gate_center).
    p_local_x = p[ri + 0] * dx + p[ri + 3] * dy + p[ri + 6] * dz
    p_local_y = p[ri + 1] * dx + p[ri + 4] * dy + p[ri + 7] * dz
    p_local_z = p[ri + 2] * dx + p[ri + 5] * dy + p[ri + 8] * dz

    oh = _GATE_FRAME_OUTER_HALF_MPC

    # Gate-frame bars. These expressions are retained so the constraint layout
    # remains unchanged, but v5 disables their bounds by default.
    h_list.append(p_local_x ** 2 + (p_local_y - oh) ** 2)
    h_list.append(p_local_x ** 2 + (p_local_y + oh) ** 2)
    h_list.append(p_local_x ** 2 + (p_local_z - oh) ** 2)
    h_list.append(p_local_x ** 2 + (p_local_z + oh) ** 2)

    # Gate funnel/opening constraints.
    h_list.append(p_local_y)
    h_list.append(p_local_z)

    con_h_expr = cs.vertcat(*h_list)

    model = AcadosModel()
    model.name = "gate_mpc_progress_safe_v5"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = x_dot
    model.con_h_expr = con_h_expr
    model.con_h_expr_e = con_h_expr

    return model


def _create_mpc_solver(
    dt: float,
    N: int,
    mass: float,
    thrust_min: float,
    thrust_max: float,
    obstacle_radius: float = 0.30,
    gate_half_opening: float = 0.16,
    gate_post_radius: float = 0.115,
    max_tilt_mpc: float = 0.45,
    az_up_max: float = 1.25,
    az_down_max: float = 1.75,
):
    ocp = AcadosOcp()
    ocp.model = _create_mpc_model()

    nx, nu = 9, 3
    ny = nx + nu
    nh = _NH

    g = 9.81

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q_base = np.diag([
        95.0, 95.0, 70.0,
        70.0, 70.0, 90.0,
        0.2, 0.2, 0.4,
    ])

    R_base = np.diag([
        0.25,
        0.25,
        0.65,
    ])

    Q_e = np.diag([
        120.0, 120.0, 80.0,
        65.0, 65.0, 90.0,
        0.1, 0.1, 0.2,
    ])

    W_base = scipy.linalg.block_diag(Q_base, R_base)

    ocp.cost.W = W_base
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)

    thrust_acc_min = thrust_min / mass
    thrust_acc_max = thrust_max / mass

    a_min_z = max(thrust_acc_min, g - az_down_max)
    a_max_z = min(thrust_acc_max, g + az_up_max)

    a_max_xy = g * math.tan(max_tilt_mpc)

    ocp.constraints.lbu = np.array([-a_max_xy, -a_max_xy, a_min_z])
    ocp.constraints.ubu = np.array([a_max_xy, a_max_xy, a_max_z])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.x0 = np.zeros(nx)

    r_sq = obstacle_radius ** 2

    lh = np.zeros(nh)
    uh = np.zeros(nh)

    lh[:_N_OBS_SLOTS] = r_sq
    uh[:_N_OBS_SLOTS] = _CONSTRAINT_OFF

    # Gate-frame bar bounds are disabled in v5. The active gate is protected by
    # the funnel/opening constraints below. This avoids artificial SQP-RTI walls.
    lh[_N_OBS_SLOTS:_TOTAL_CYL] = -_CONSTRAINT_OFF
    uh[_N_OBS_SLOTS:_TOTAL_CYL] = _CONSTRAINT_OFF

    lh[_TOTAL_CYL:] = -_CONSTRAINT_OFF
    uh[_TOTAL_CYL:] = _CONSTRAINT_OFF

    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.idxsh = np.arange(nh)

    zl = np.zeros(nh)
    zu = np.zeros(nh)
    Zl = np.zeros(nh)
    Zu = np.zeros(nh)

    # Obstacle cylinders: expensive but not so huge that the solver chooses to
    # hover when a first-gate corridor is tight.
    zl[:_N_OBS_SLOTS] = 8.0e3
    Zl[:_N_OBS_SLOTS] = 8.0e4

    # Gate opening/funnel: strong guidance, but softer than obstacle avoidance.
    # If the drone is slightly misaligned, it should still pass through rather
    # than freeze in front of the gate.
    zl[_TOTAL_CYL:] = 2.5e3
    zu[_TOTAL_CYL:] = 2.5e3
    Zl[_TOTAL_CYL:] = 2.5e4
    Zu[_TOTAL_CYL:] = 2.5e4

    ocp.cost.zl = zl
    ocp.cost.zu = zu
    ocp.cost.Zl = Zl
    ocp.cost.Zu = Zu

    ocp.constraints.lh_e = lh.copy()
    ocp.constraints.uh_e = uh.copy()
    ocp.constraints.idxsh_e = np.arange(nh)
    ocp.cost.zl_e = zl.copy()
    ocp.cost.zu_e = zu.copy()
    ocp.cost.Zl_e = Zl.copy()
    ocp.cost.Zu_e = Zu.copy()

    p0 = np.zeros(_N_PARAMS)

    for i in range(_TOTAL_CYL):
        p0[2 * i] = 100.0
        p0[2 * i + 1] = 100.0

    gi = _TOTAL_CYL * 2
    p0[gi:gi + 3] = [100.0, 100.0, 100.0]
    p0[gi + 3:gi + 12] = np.eye(3).flatten()

    ocp.parameter_values = p0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver_cond_N = min(N, 10)
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 30
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.tf = dt * N

    code_dir = Path(__file__).parent / "c_generated_code"
    code_dir.mkdir(exist_ok=True)

    solver = AcadosOcpSolver(
        ocp,
        json_file=str(code_dir / "gate_mpc_progress_safe_v5.json"),
        verbose=False,
        build=True,
        generate=True,
    )

    gamma = 0.96
    weights = np.array([gamma ** i for i in range(N)])
    weights *= N / weights.sum()

    for i in range(N):
        solver.cost_set(i, "W", weights[i] * W_base)

    solver.cost_set(N, "W", Q_e)

    return solver, nx, nu, W_base, Q_e


def _vertical_preserving_thrust(
    az_world: float,
    roll: float,
    pitch: float,
    mass: float,
    thrust_min: float,
    thrust_max: float,
) -> float:
    """Compute thrust so vertical component equals desired world z specific thrust."""
    c = math.cos(roll) * math.cos(pitch)
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
    """Convert world-frame specific thrust vector to attitude + total thrust."""
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
    pitch = float(np.arctan2(zx_local, np.sqrt(zy_local ** 2 + zz_local ** 2)))

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
    """Gate-target MPC controller for drone racing."""

    OBSTACLE_RADIUS = 0.25
    OBSTACLE_MPC_MARGIN = 0.05
    OBSTACLE_ROUTE_MARGIN = 0.14

    GATE_HALF_OPENING = 0.16
    GATE_OUTER_HALF = _GATE_FRAME_OUTER_HALF_MPC
    GATE_POST_RADIUS = 0.115

    FUNNEL_LENGTH = 0.60
    FUNNEL_OUTER_HALF = 0.42

    APPROACH_DIST = 0.50
    APPROACH_DIST_MIN = 0.12
    EXIT_DIST = 0.34

    ROUTE_DETOUR_EXTRA = 0.18

    GROUND_CLEARANCE = 0.10
    CEILING = 1.80

    V_CRUISE = 1.30
    V_GATE = 0.95

    MPC_N = 50
    MPC_DT = 0.05

    APF_INFLUENCE = 0.55
    APF_GAIN = 0.20
    APF_MAX = 0.95

    GATE_PUSH_DIST = 0.80
    GATE_PUSH_GAIN = 1.35
    GATE_PUSH_MAX = 1.10

    ALIGN_START_DIST = 1.20

    MAX_TILT_CMD = 0.50
    MAX_TILT_MPC = 0.45

    AZ_UP_MAX = 1.25
    AZ_DOWN_MAX = 1.75

    VZ_REF_MAX = 0.40
    LAUNCH_HOLD_TIME = 0.20
    LAUNCH_BLEND_TIME = 0.55

    Z_HOLD_KP = 2.8
    Z_HOLD_KD = 2.0

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._g = 9.81
        self._dt = 1.0 / config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)

        self._mass = float(drone_params["mass"])
        self._thrust_min = float(drone_params["thrust_min"]) * 4
        self._thrust_max = float(drone_params["thrust_max"]) * 4

        self._gate_positions = np.array([g.tolist() for g in obs["gates_pos"]], dtype=np.float64)
        self._gate_quats = np.array([g.tolist() for g in obs["gates_quat"]], dtype=np.float64)
        self._gate_rotmats = [Rot.from_quat(q).as_matrix() for q in self._gate_quats]

        self._n_gates = len(self._gate_positions)
        self._target_gate = int(obs["target_gate"])

        self._initial_pos = np.asarray(obs["pos"], dtype=np.float64).copy()
        self._gate_plan_rotmats: list[np.ndarray] = []
        self._refresh_gate_frames()

        self._obstacle_positions = np.array(
            [g.tolist() for g in obs["obstacles_pos"]],
            dtype=np.float64,
        )

        self._gates_visited = obs["gates_visited"].copy()

        self._obstacles_visited = np.array(
            obs.get(
                "obstacles_visited",
                np.zeros(len(self._obstacle_positions), dtype=bool),
            ),
            dtype=bool,
        )

        print("[GATE-MPC] Building MPC solver...")

        self._solver, self._nx, self._nu, self._W_base, self._Q_e = _create_mpc_solver(
            dt=self.MPC_DT,
            N=self.MPC_N,
            mass=self._mass,
            thrust_min=self._thrust_min,
            thrust_max=self._thrust_max,
            obstacle_radius=self.OBSTACLE_RADIUS + self.OBSTACLE_MPC_MARGIN,
            gate_half_opening=self.GATE_HALF_OPENING,
            gate_post_radius=self.GATE_POST_RADIUS,
            max_tilt_mpc=self.MAX_TILT_MPC,
            az_up_max=self.AZ_UP_MAX,
            az_down_max=self.AZ_DOWN_MAX,
        )

        self._nh = _NH
        self._n_params = _N_PARAMS

        print(
            f"[GATE-MPC] Solver ready. N={self.MPC_N}, dt={self.MPC_DT:.3f}s, "
            f"horizon={self.MPC_N * self.MPC_DT:.1f}s"
        )

        self._prev_accel = np.array([0.0, 0.0, self._g], dtype=np.float64)
        self._prev_output = np.array([0.0, 0.0, 0.0, self._mass * self._g], dtype=np.float64)

        self._tick = 0
        self._finished = False

        self._mass_estimate = self._mass

        self._launch_z: float | None = None
        self._last_pos = np.zeros(3, dtype=np.float64)
        self._last_ref_pos: np.ndarray | None = None

    def _refresh_gate_frames(self) -> None:
        """Cache one planning frame per gate.

        The previous dynamic normal selection flipped direction after crossing
        the gate plane if the environment had not marked the gate as visited yet.
        That made the controller point back toward gate 0 and hover/oscillate.
        """
        self._gate_plan_rotmats = []

        for gi in range(self._n_gates):
            R = self._gate_rotmats[gi].copy()
            normal = R[:, 0].copy()

            if gi == 0:
                from_ref = self._initial_pos
            else:
                prev_R = self._gate_plan_rotmats[gi - 1]
                from_ref = self._gate_positions[gi - 1] + self.EXIT_DIST * prev_R[:, 0]

            to_gate = self._gate_positions[gi] - from_ref

            if float(np.dot(to_gate, normal)) < 0.0:
                R[:, 0] = -normal

            self._gate_plan_rotmats.append(R)

    def _get_gate_normal(self, gi: int, from_pos: np.ndarray | None = None) -> np.ndarray:
        """Get the cached planning normal for a gate."""
        del from_pos
        if 0 <= gi < len(self._gate_plan_rotmats):
            return self._gate_plan_rotmats[gi][:, 0].copy()
        return self._gate_rotmats[gi][:, 0].copy()

    def _get_gate_frame(self, gi: int, from_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return cached gate normal and planning frame.

        from_pos is accepted for compatibility but intentionally ignored.
        """
        del from_pos
        if 0 <= gi < len(self._gate_plan_rotmats):
            R = self._gate_plan_rotmats[gi].copy()
        else:
            R = self._gate_rotmats[gi].copy()
        return R[:, 0].copy(), R

    @staticmethod
    def _append_waypoint(
        waypoints: list[np.ndarray],
        point: np.ndarray,
        min_sep: float = 0.04,
    ) -> None:
        p = np.asarray(point, dtype=np.float64)

        if not waypoints:
            waypoints.append(p.copy())
            return

        if float(np.linalg.norm(p - waypoints[-1])) > min_sep:
            waypoints.append(p.copy())

    @staticmethod
    def _segment_distance_xy(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """Distance from point p to segment a-b in XY."""
        a2 = np.asarray(a[:2], dtype=np.float64)
        b2 = np.asarray(b[:2], dtype=np.float64)
        p2 = np.asarray(p[:2], dtype=np.float64)

        ab = b2 - a2
        ab2 = float(np.dot(ab, ab))

        if ab2 < _EPS:
            return float(np.linalg.norm(p2 - a2)), 0.0, a2.copy()

        t = float(np.clip(np.dot(p2 - a2, ab) / ab2, 0.0, 1.0))
        closest = a2 + t * ab

        return float(np.linalg.norm(p2 - closest)), t, closest

    def _find_first_blocking_obstacle(
        self,
        a: np.ndarray,
        b: np.ndarray,
        clearance: float | None = None,
    ) -> tuple[np.ndarray, float, float] | None:
        """Find the first obstacle that blocks segment a-b in XY."""
        if len(self._obstacle_positions) == 0:
            return None

        if clearance is None:
            clearance = self.OBSTACLE_RADIUS + self.OBSTACLE_ROUTE_MARGIN

        clearance = float(clearance)
        best: tuple[np.ndarray, float, float] | None = None

        for op in self._obstacle_positions:
            d, t, _ = self._segment_distance_xy(a, b, op)

            if 0.03 < t < 0.97 and d < clearance:
                if best is None or t < best[2]:
                    best = (op.copy(), d, t)

        return best

    def _detour_point_around_obstacle(
        self,
        a: np.ndarray,
        b: np.ndarray,
        obstacle: np.ndarray,
        clearance: float,
    ) -> np.ndarray | None:
        """Create a lateral detour point around one blocking obstacle."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        obstacle = np.asarray(obstacle, dtype=np.float64)

        dxy = b[:2] - a[:2]
        seg_len = float(np.linalg.norm(dxy))

        if seg_len < _EPS:
            return None

        direction = dxy / seg_len
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)

        _, t, _ = self._segment_distance_xy(a, b, obstacle)

        z = float((1.0 - t) * a[2] + t * b[2])
        z = float(np.clip(z, self.GROUND_CLEARANCE + 0.05, self.CEILING - 0.05))

        offset = float(clearance + self.ROUTE_DETOUR_EXTRA)
        candidates: list[tuple[float, np.ndarray]] = []

        for side in (-1.0, 1.0):
            c2 = obstacle[:2] + side * offset * perp
            cand = np.array([c2[0], c2[1], z], dtype=np.float64)

            score = float(np.linalg.norm(cand - a) + np.linalg.norm(b - cand))

            min_clear = 1e9
            for other in self._obstacle_positions:
                d1, _, _ = self._segment_distance_xy(a, cand, other)
                d2, _, _ = self._segment_distance_xy(cand, b, other)
                min_clear = min(min_clear, d1, d2)

            if min_clear < clearance:
                score += 1000.0 * float((clearance - min_clear) ** 2)

            candidates.append((score, cand))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])[1]

    def _append_clear_segment(
        self,
        waypoints: list[np.ndarray],
        end: np.ndarray,
        clearance: float | None = None,
        max_detours: int = 2,
    ) -> None:
        """Append end to waypoints, inserting detours if the segment is blocked."""
        if clearance is None:
            clearance = self.OBSTACLE_RADIUS + self.OBSTACLE_ROUTE_MARGIN

        end = np.asarray(end, dtype=np.float64)
        start = waypoints[-1]

        block = self._find_first_blocking_obstacle(start, end, clearance)

        if block is not None and max_detours > 0:
            obstacle, _, _ = block
            detour = self._detour_point_around_obstacle(start, end, obstacle, clearance)

            if (
                detour is not None
                and float(np.linalg.norm(detour - start)) > 0.08
                and float(np.linalg.norm(detour - end)) > 0.08
            ):
                self._append_clear_segment(
                    waypoints,
                    detour,
                    clearance=clearance,
                    max_detours=max_detours - 1,
                )
                self._append_clear_segment(
                    waypoints,
                    end,
                    clearance=clearance,
                    max_detours=max_detours - 1,
                )
                return

        self._append_waypoint(waypoints, end)

    def _choose_approach_point(
        self,
        gi: int,
        from_pos: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        """Choose a gate approach point.

        Use a straight approach when the corridor is clear. If a pole blocks the
        long final approach, shorten the approach so the route can go around the
        pole first and still enter the gate from the correct side.
        """
        del from_pos
        gp = self._gate_positions[gi]

        approach_distances = [
            self.APPROACH_DIST,
            0.75 * self.APPROACH_DIST,
            0.55 * self.APPROACH_DIST,
            self.APPROACH_DIST_MIN,
        ]

        approach_clearance = self.OBSTACLE_RADIUS + 0.045

        for d in approach_distances:
            d = max(float(d), self.APPROACH_DIST_MIN)
            approach = gp - d * normal

            if self._find_first_blocking_obstacle(
                approach,
                gp,
                clearance=approach_clearance,
            ) is None:
                return approach

        return gp - self.APPROACH_DIST_MIN * normal

    def _select_active_obstacles(
        self,
        pos: np.ndarray,
        ref_pos: np.ndarray,
    ) -> np.ndarray:
        """Pick the obstacle slots that matter for the current MPC horizon."""
        if len(self._obstacle_positions) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        sample_count = min(14, len(ref_pos))
        sample_idx = np.linspace(0, len(ref_pos) - 1, sample_count, dtype=int)
        samples_xy = ref_pos[sample_idx, :2]

        scores: list[tuple[float, int]] = []

        for oi, op in enumerate(self._obstacle_positions):
            d_now = float(np.linalg.norm(pos[:2] - op[:2]))
            d_path = float(np.min(np.linalg.norm(samples_xy - op[:2], axis=1)))
            scores.append((min(d_now, d_path) + 1e-3 * oi, oi))

        order = [oi for _, oi in sorted(scores)[:_N_OBS_SLOTS]]

        return self._obstacle_positions[order].copy()

    def _avoidance_accel(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Acceleration-level local collision avoidance for poles."""
        acc = np.zeros(3, dtype=np.float64)

        if len(self._obstacle_positions) == 0:
            return acc

        for op in self._obstacle_positions:
            diff = pos[:2] - op[:2]
            d = float(np.linalg.norm(diff))

            if not (_EPS < d < self.APF_INFLUENCE):
                continue

            away = diff / d
            closing_speed = max(0.0, -float(np.dot(vel[:2], away)))

            barrier = self.APF_GAIN * (1.0 / d - 1.0 / self.APF_INFLUENCE) / (d * d)
            mag = min(barrier + 0.65 * closing_speed, self.APF_MAX)

            acc[:2] += mag * away

        acc_norm = float(np.linalg.norm(acc[:2]))

        if acc_norm > self.APF_MAX:
            acc[:2] *= self.APF_MAX / acc_norm

        return acc

    def _gate_progress_accel(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Small acceleration along the gate normal when already in the corridor."""
        acc = np.zeros(3, dtype=np.float64)

        if not (0 <= self._target_gate < self._n_gates):
            return acc

        gp = self._gate_positions[self._target_gate]
        normal, R = self._get_gate_frame(self._target_gate)
        local = R.T @ (pos - gp)

        # Only help when near the active gate and approximately inside the
        # broad funnel. Outside this region, the route planner and obstacle
        # avoidance should decide the path.
        if local[0] > self.EXIT_DIST:
            return acc

        if abs(float(local[0])) > self.GATE_PUSH_DIST:
            return acc

        if abs(float(local[1])) > self.FUNNEL_OUTER_HALF + 0.08:
            return acc

        if abs(float(local[2])) > self.FUNNEL_OUTER_HALF + 0.08:
            return acc

        nxy = normal.copy()
        nxy[2] = 0.0
        nxy_norm = float(np.linalg.norm(nxy))

        if nxy_norm < _EPS:
            return acc

        nxy /= nxy_norm

        vn = float(np.dot(vel[:2], nxy[:2]))
        desired_vn = self.V_GATE
        mag = float(np.clip(self.GATE_PUSH_GAIN * (desired_vn - vn), 0.0, self.GATE_PUSH_MAX))

        acc[:2] = mag * nxy[:2]
        return acc

    def _apply_launch_altitude_ramp(self, ref_pos: np.ndarray, ref_vel: np.ndarray) -> None:
        """Prevent the first MPC horizon from immediately pulling upward."""
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

    def _generate_references(self, pos: np.ndarray, vel: np.ndarray):
        """Generate MPC reference positions and velocities.

        The velocity reference follows the route tangent. Near or just after the
        current gate plane, the route targets the exit point instead of turning
        back to the gate center.
        """
        ref_pos = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)
        ref_vel = np.zeros((self.MPC_N + 1, 3), dtype=np.float64)

        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            ref_pos[:] = pos
            return ref_pos, ref_vel

        gp = self._gate_positions[self._target_gate]
        normal, R_gate = self._get_gate_frame(self._target_gate)
        local_pos = R_gate.T @ (pos - gp)

        approach_pt = self._choose_approach_point(self._target_gate, pos, normal)
        exit_pt = gp + self.EXIT_DIST * normal

        waypoints: list[np.ndarray] = [pos.copy()]
        route_clearance = self.OBSTACLE_RADIUS + self.OBSTACLE_ROUTE_MARGIN

        before_gate = float(local_pos[0]) < -0.06
        still_need_exit = float(local_pos[0]) < 1.20 * self.EXIT_DIST

        if before_gate:
            dist_to_approach = float(np.linalg.norm(pos - approach_pt))

            if dist_to_approach > 0.12:
                self._append_clear_segment(
                    waypoints,
                    approach_pt,
                    clearance=route_clearance,
                    max_detours=2,
                )

            # Final pass through gate. The chosen approach point should make this
            # segment clear. Do not detour here, otherwise the controller can cut
            # into the frame.
            self._append_clear_segment(
                waypoints,
                gp.copy(),
                clearance=route_clearance,
                max_detours=0,
            )

        if still_need_exit:
            self._append_clear_segment(
                waypoints,
                exit_pt,
                clearance=route_clearance,
                max_detours=0,
            )

        next_gate = self._target_gate + 1
        has_next = next_gate < self._n_gates

        if has_next:
            next_gp = self._gate_positions[next_gate]
            next_normal, _ = self._get_gate_frame(next_gate)
            next_approach = self._choose_approach_point(next_gate, exit_pt, next_normal)

            self._append_clear_segment(
                waypoints,
                next_approach,
                clearance=route_clearance,
                max_detours=2,
            )

            self._append_clear_segment(
                waypoints,
                next_gp.copy(),
                clearance=route_clearance,
                max_detours=0,
            )

        if len(waypoints) < 2:
            ref_pos[:] = pos
            self._apply_launch_altitude_ramp(ref_pos, ref_vel)
            return ref_pos, ref_vel

        waypoints_arr = [np.asarray(w, dtype=np.float64) for w in waypoints]

        cum_dist = [0.0]

        for i in range(1, len(waypoints_arr)):
            seg_len = float(np.linalg.norm(waypoints_arr[i] - waypoints_arr[i - 1]))
            cum_dist.append(cum_dist[-1] + seg_len)

        total_dist = cum_dist[-1]

        if total_dist < _EPS:
            ref_pos[:] = pos
            self._apply_launch_altitude_ramp(ref_pos, ref_vel)
            return ref_pos, ref_vel

        speed_now = float(np.linalg.norm(vel))
        avg_speed = min(self.V_CRUISE, max(0.65, 0.5 * (speed_now + self.V_CRUISE)))

        nominal_wp_count = 3 + (2 if has_next else 0)
        if len(waypoints_arr) > nominal_wp_count:
            avg_speed = min(avg_speed, 1.10)

        for i in range(self.MPC_N + 1):
            t_hor = i * self.MPC_DT
            s = min(avg_speed * t_hor, total_dist)

            seg_idx = len(cum_dist) - 2
            for j in range(len(cum_dist) - 1):
                if s <= cum_dist[j + 1] or j == len(cum_dist) - 2:
                    seg_idx = j
                    break

            seg_len = cum_dist[seg_idx + 1] - cum_dist[seg_idx]
            if seg_len > _EPS:
                alpha = float(np.clip((s - cum_dist[seg_idx]) / seg_len, 0.0, 1.0))
            else:
                alpha = 0.0

            a = waypoints_arr[seg_idx]
            b = waypoints_arr[seg_idx + 1]

            ref_pos[i] = (1.0 - alpha) * a + alpha * b
            ref_pos[i, 2] = float(np.clip(ref_pos[i, 2], self.GROUND_CLEARANCE, self.CEILING))

            seg_vec = b - a
            seg_norm = float(np.linalg.norm(seg_vec))
            if seg_norm > _EPS:
                route_dir = seg_vec / seg_norm
            else:
                route_dir = np.zeros(3, dtype=np.float64)

            speed_ref = self.V_CRUISE

            near_gate_dist = float(np.linalg.norm(ref_pos[i] - gp))
            if has_next:
                near_gate_dist = min(
                    near_gate_dist,
                    float(np.linalg.norm(ref_pos[i] - self._gate_positions[next_gate])),
                )

            if near_gate_dist < self.ALIGN_START_DIST:
                beta = float(np.clip(near_gate_dist / self.ALIGN_START_DIST, 0.0, 1.0))
                speed_ref = min(speed_ref, self.V_GATE + (self.V_CRUISE - self.V_GATE) * beta)

            if seg_idx < len(waypoints_arr) - 2:
                next_vec = waypoints_arr[seg_idx + 2] - waypoints_arr[seg_idx + 1]
                next_norm = float(np.linalg.norm(next_vec))
                remaining_to_corner = max(cum_dist[seg_idx + 1] - s, 0.0)

                if seg_norm > _EPS and next_norm > _EPS and remaining_to_corner < 0.55:
                    next_dir = next_vec / next_norm
                    cos_turn = float(np.clip(np.dot(route_dir, next_dir), -1.0, 1.0))
                    turn_angle = math.acos(cos_turn)

                    corner_blend = 1.0 - remaining_to_corner / 0.55
                    turn_factor = 1.0 - 0.25 * corner_blend * (turn_angle / math.pi)
                    turn_factor = float(np.clip(turn_factor, 0.70, 1.0))
                    speed_ref *= turn_factor

            ref_vel[i] = route_dir * speed_ref
            ref_vel[i, 2] = float(np.clip(ref_vel[i, 2], -self.VZ_REF_MAX, self.VZ_REF_MAX))

        self._apply_launch_altitude_ramp(ref_pos, ref_vel)

        return ref_pos, ref_vel

    def _limit_commanded_accel(self, u: np.ndarray) -> np.ndarray:
        """Final safety limit before converting acceleration to attitude."""
        u_limited = np.asarray(u, dtype=np.float64).copy()

        a_xy_max = self._g * math.tan(self.MAX_TILT_MPC)

        u_limited[0] = float(np.clip(u_limited[0], -a_xy_max, a_xy_max))
        u_limited[1] = float(np.clip(u_limited[1], -a_xy_max, a_xy_max))

        u_limited[2] = float(
            np.clip(
                u_limited[2],
                self._g - self.AZ_DOWN_MAX,
                self._g + self.AZ_UP_MAX,
            )
        )

        return u_limited

    def _launch_altitude_guard(self, u0: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Blend vertical command with launch-altitude hold during first moments."""
        if self._launch_z is None:
            return u0

        elapsed = max(0.0, (self._tick - 1) * self._dt)

        if elapsed >= self.LAUNCH_BLEND_TIME:
            return u0

        z_err = self._launch_z - float(pos[2])
        vz_err = -float(vel[2])

        az_hold = self._g + self.Z_HOLD_KP * z_err + self.Z_HOLD_KD * vz_err
        az_hold = float(np.clip(az_hold, self._g - self.AZ_DOWN_MAX, self._g + self.AZ_UP_MAX))

        alpha = 1.0 - elapsed / max(self.LAUNCH_BLEND_TIME, _EPS)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        u_guarded = u0.copy()
        u_guarded[2] = alpha * az_hold + (1.0 - alpha) * u0[2]

        return u_guarded

    def _warm_start_solver(self, x0: np.ndarray, ref_pos: np.ndarray, ref_vel: np.ndarray) -> None:
        """Warm-start SQP-RTI around the intended route.

        This is important for nonconvex obstacle constraints. Without this, the
        RTI linearization can stay around an old hovering solution and block
        progress through the first gate.
        """
        hover = np.array([0.0, 0.0, self._g], dtype=np.float64)

        for i in range(self.MPC_N + 1):
            if i == 0:
                x_guess = x0.copy()
            else:
                x_guess = np.zeros(self._nx, dtype=np.float64)
                x_guess[0:3] = ref_pos[i]
                x_guess[3:6] = ref_vel[i]
                x_guess[6:9] = hover

            self._solver.set(i, "x", x_guess)

        for i in range(self.MPC_N):
            u_guess = hover.copy()
            if i == 0:
                u_guess = 0.65 * self._prev_accel + 0.35 * hover
            self._solver.set(i, "u", self._limit_commanded_accel(u_guess))

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        self._tick += 1

        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)

        self._last_pos = pos.copy()

        if self._launch_z is None:
            self._launch_z = float(pos[2])

        if self._finished:
            return np.array([0.0, 0.0, 0.0, self._mass * self._g], dtype=np.float32)

        ref_pos, ref_vel = self._generate_references(pos, vel)
        self._last_ref_pos = ref_pos.copy()

        x0 = np.concatenate([pos, vel, self._prev_accel])

        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        params = np.zeros(self._n_params, dtype=np.float64)

        active_obstacles = self._select_active_obstacles(pos, ref_pos)

        for oi in range(_N_OBS_SLOTS):
            if oi < len(active_obstacles):
                params[2 * oi] = active_obstacles[oi][0]
                params[2 * oi + 1] = active_obstacles[oi][1]
            else:
                params[2 * oi] = 100.0
                params[2 * oi + 1] = 100.0

        # Legacy slots for disabled gate-frame bar expressions.
        for pi in range(_N_OBS_SLOTS, _TOTAL_CYL):
            params[2 * pi] = 100.0
            params[2 * pi + 1] = 100.0

        gi_p = _TOTAL_CYL * 2

        if 0 <= self._target_gate < self._n_gates:
            gp_cur = self._gate_positions[self._target_gate]
            _, R_cur = self._get_gate_frame(self._target_gate)
        else:
            gp_cur = np.array([100.0, 100.0, 100.0], dtype=np.float64)
            R_cur = np.eye(3)

        next_gate_idx = self._target_gate + 1
        has_next = 0 <= self._target_gate < self._n_gates and next_gate_idx < self._n_gates

        if has_next:
            gp_next = self._gate_positions[next_gate_idx]
            _, R_next = self._get_gate_frame(next_gate_idx)
        else:
            gp_next = None
            R_next = None

        for i in range(self.MPC_N + 1):
            use_next = False

            if 0 <= self._target_gate < self._n_gates:
                p_local_cur = R_cur.T @ (ref_pos[i] - gp_cur)
                if p_local_cur[0] > self.EXIT_DIST and has_next:
                    use_next = True

            params_i = params.copy()

            if use_next:
                params_i[gi_p:gi_p + 3] = gp_next
                params_i[gi_p + 3:gi_p + 12] = R_next.flatten()
            else:
                params_i[gi_p:gi_p + 3] = gp_cur
                params_i[gi_p + 3:gi_p + 12] = R_cur.flatten()

            self._solver.set(i, "p", params_i)

        r_sq = (self.OBSTACLE_RADIUS + self.OBSTACLE_MPC_MARGIN) ** 2
        h_open = self.GATE_HALF_OPENING

        for i in range(1, self.MPC_N + 1):
            lh_i = np.zeros(self._nh, dtype=np.float64)
            uh_i = np.zeros(self._nh, dtype=np.float64)

            lh_i[:_N_OBS_SLOTS] = r_sq
            uh_i[:_N_OBS_SLOTS] = _CONSTRAINT_OFF

            # Disabled gate-frame bars. The active gate frame is handled by the
            # convex-ish y/z funnel constraints below.
            lh_i[_N_OBS_SLOTS:_TOTAL_CYL] = -_CONSTRAINT_OFF
            uh_i[_N_OBS_SLOTS:_TOTAL_CYL] = _CONSTRAINT_OFF

            if 0 <= self._target_gate < self._n_gates:
                p_local_cur = R_cur.T @ (ref_pos[i] - gp_cur)
                use_next = p_local_cur[0] > self.EXIT_DIST and has_next

                if use_next:
                    gate_pos_i = gp_next
                    R_i = R_next
                else:
                    gate_pos_i = gp_cur
                    R_i = R_cur

                p_local_i = R_i.T @ (ref_pos[i] - gate_pos_i)
                dist_to_plane = abs(float(p_local_i[0]))

                if dist_to_plane < self.FUNNEL_LENGTH:
                    alpha = dist_to_plane / max(self.FUNNEL_LENGTH, _EPS)
                    h_bound = h_open + alpha * (self.FUNNEL_OUTER_HALF - h_open)

                    lh_i[_TOTAL_CYL] = -h_bound
                    uh_i[_TOTAL_CYL] = h_bound
                    lh_i[_TOTAL_CYL + 1] = -h_bound
                    uh_i[_TOTAL_CYL + 1] = h_bound
                else:
                    lh_i[_TOTAL_CYL:] = -_CONSTRAINT_OFF
                    uh_i[_TOTAL_CYL:] = _CONSTRAINT_OFF
            else:
                lh_i[_TOTAL_CYL:] = -_CONSTRAINT_OFF
                uh_i[_TOTAL_CYL:] = _CONSTRAINT_OFF

            if i < self.MPC_N:
                self._solver.constraints_set(i, "lh", lh_i)
                self._solver.constraints_set(i, "uh", uh_i)
            else:
                self._solver.constraints_set(self.MPC_N, "lh", lh_i)
                self._solver.constraints_set(self.MPC_N, "uh", uh_i)

        gamma = 0.96
        weights_raw = np.array([gamma ** k for k in range(self.MPC_N)])
        weights_raw *= self.MPC_N / weights_raw.sum()

        for i in range(self.MPC_N):
            yref = np.zeros(self._nx + self._nu, dtype=np.float64)

            yref[0:3] = ref_pos[i]
            yref[3:6] = ref_vel[i]
            yref[6:9] = np.array([0.0, 0.0, self._g])
            yref[9:12] = np.array([0.0, 0.0, self._g])

            self._solver.set(i, "yref", yref)

            W_i = weights_raw[i] * self._W_base

            if 0 <= self._target_gate < self._n_gates:
                gate_dist = float(np.linalg.norm(ref_pos[i] - gp_cur))

                if gate_dist < self.ALIGN_START_DIST:
                    boost = 1.0 + 2.5 * (1.0 - gate_dist / self.ALIGN_START_DIST)

                    W_i = W_i.copy()
                    W_i[3, 3] *= boost
                    W_i[4, 4] *= boost
                    W_i[5, 5] *= 0.8 * boost

            self._solver.cost_set(i, "W", W_i)

        yref_e = np.zeros(self._nx, dtype=np.float64)
        yref_e[0:3] = ref_pos[self.MPC_N]
        yref_e[3:6] = ref_vel[self.MPC_N]
        yref_e[6:9] = np.array([0.0, 0.0, self._g])

        self._solver.set(self.MPC_N, "yref", yref_e)

        self._warm_start_solver(x0, ref_pos, ref_vel)

        status = self._solver.solve()

        if status <= 2:
            u0 = np.asarray(self._solver.get(0, "u"), dtype=np.float64)
        else:
            pull = ref_pos[min(8, self.MPC_N)] - pos
            pull_dist = float(np.linalg.norm(pull))

            if pull_dist > 0.01:
                direction = pull / pull_dist
                u0 = direction * min(3.2, pull_dist * 5.0)
                u0[2] += self._g
            else:
                u0 = np.array([0.0, 0.0, self._g], dtype=np.float64)

        u0 = self._limit_commanded_accel(u0)
        u0 = self._launch_altitude_guard(u0, pos, vel)

        # Progress first, then local obstacle shielding. Both are in world-frame
        # acceleration coordinates before attitude conversion.
        u0[:2] += self._gate_progress_accel(pos, vel)[:2]
        u0[:2] += self._avoidance_accel(pos, vel)[:2]

        u0 = self._limit_commanded_accel(u0)

        self._prev_accel = u0.copy()

        if self._tick % 5 == 0:
            speed = float(np.linalg.norm(vel))
            vz = float(vel[2])

            if 0 <= self._target_gate < self._n_gates:
                gate_dist = float(np.linalg.norm(pos - self._gate_positions[self._target_gate]))
                gp_t = self._gate_positions[self._target_gate]
                dz = float(gp_t[2] - pos[2])
                _, R_dbg = self._get_gate_frame(self._target_gate)
                local_dbg = R_dbg.T @ (pos - gp_t)
                gate_x = float(local_dbg[0])
            else:
                gate_dist = -1.0
                dz = 0.0
                gate_x = 0.0

            print(
                f"[GATE-MPC] step={self._tick} "
                f"pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] "
                f"v={speed:.2f} vz={vz:.2f} dz={dz:.2f} "
                f"u=[{u0[0]:.2f},{u0[1]:.2f},{u0[2]:.2f}] "
                f"gate_dist={gate_dist:.3f} gate_x={gate_x:.3f} "
                f"gate={self._target_gate} status={status}"
            )

        current_rpy = Rot.from_quat(obs["quat"]).as_euler("xyz")
        current_yaw = float(current_rpy[2])

        if 0 <= self._target_gate < self._n_gates:
            to_gate = self._gate_positions[self._target_gate] - pos

            if np.linalg.norm(to_gate[:2]) > 0.1:
                yaw_des = float(np.arctan2(to_gate[1], to_gate[0]))
            else:
                yaw_des = current_yaw
        else:
            yaw_des = current_yaw

        yaw_error = yaw_des - current_yaw
        yaw_error = (yaw_error + np.pi) % (2.0 * np.pi) - np.pi

        yaw_cmd = current_yaw + float(np.clip(0.15 * yaw_error, -0.1, 0.1))

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

        thrust_cmd = _vertical_preserving_thrust(
            az_world=float(u0[2]),
            roll=roll_cmd,
            pitch=pitch_cmd,
            mass=self._mass_estimate,
            thrust_min=self._thrust_min,
            thrust_max=self._thrust_max,
        )

        z = float(pos[2])

        if z < self.GROUND_CLEARANCE + 0.1:
            thrust_cmd += (
                0.4
                * (self.GROUND_CLEARANCE + 0.1 - z)
                * self._mass_estimate
                * self._g
            )

        if z > self.CEILING - 0.2:
            overshoot = z - (self.CEILING - 0.2)
            thrust_cmd -= (
                (0.5 + 2.0 * overshoot)
                * overshoot
                * self._mass_estimate
                * self._g
            )

        thrust_cmd = float(np.clip(thrust_cmd, self._thrust_min, self._thrust_max))

        self._prev_output = np.array([roll_cmd, pitch_cmd, yaw_out, thrust_cmd], dtype=np.float64)

        return self._prev_output.astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        del reward, terminated, truncated, info

        new_target = int(obs["target_gate"])
        gate_changed = new_target != self._target_gate

        self._target_gate = new_target

        if new_target < 0:
            self._finished = True
            return True

        frames_changed = False

        for i in range(self._n_gates):
            if obs["gates_visited"][i] and not self._gates_visited[i]:
                self._gates_visited[i] = True
                self._gate_positions[i] = np.array(obs["gates_pos"][i], dtype=np.float64)
                self._gate_quats[i] = np.array(obs["gates_quat"][i], dtype=np.float64)
                self._gate_rotmats[i] = Rot.from_quat(self._gate_quats[i]).as_matrix()
                frames_changed = True

        if frames_changed or gate_changed:
            self._refresh_gate_frames()

        for i in range(len(self._obstacle_positions)):
            if obs["obstacles_visited"][i] and not self._obstacles_visited[i]:
                self._obstacles_visited[i] = True
                self._obstacle_positions[i] = np.array(obs["obstacles_pos"][i], dtype=np.float64)

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

    def render_callback(self, sim: Sim):
        from crazyflow.sim.visualize import draw_line, draw_points

        drone_pos = getattr(self, "_last_pos", np.zeros(3))

        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]

            draw_points(
                sim,
                gp.reshape(1, -1),
                rgba=(1.0, 1.0, 0.0, 1.0),
                size=0.04,
            )

            draw_line(
                sim,
                np.vstack([drone_pos, gp]),
                rgba=(0.0, 1.0, 0.0, 0.6),
                start_size=2.0,
                end_size=2.0,
            )

        # Yellow line: the MPC reference trajectory being tracked. If it bends
        # around a pole, the MPC is being asked to go around it. If it goes
        # through a pole, the route generator is the issue.
        ref_path = getattr(self, "_last_ref_pos", None)

        if ref_path is not None and len(ref_path) > 1:
            draw_line(
                sim,
                ref_path[::2],
                rgba=(1.0, 1.0, 0.0, 0.65),
                start_size=2.0,
                end_size=2.0,
            )

        VIS_RANGE = 1.5

        for op in self._obstacle_positions:
            if np.linalg.norm(drone_pos[:2] - op[:2]) > VIS_RANGE:
                continue

            n_ring = 24
            angles = np.linspace(0, 2.0 * np.pi, n_ring, endpoint=False)

            ring_pts = np.zeros((n_ring, 3), dtype=np.float64)

            for j in range(n_ring):
                ring_pts[j, 0] = op[0] + self.OBSTACLE_RADIUS * np.cos(angles[j])
                ring_pts[j, 1] = op[1] + self.OBSTACLE_RADIUS * np.sin(angles[j])
                ring_pts[j, 2] = 0.5

            ring_closed = np.vstack([ring_pts, ring_pts[0:1]])

            draw_line(
                sim,
                ring_closed,
                rgba=(1.0, 0.0, 0.0, 0.7),
                start_size=2.0,
                end_size=2.0,
            )

            for z_h in [0.2, 0.8, 1.2]:
                ring_h = ring_closed.copy()
                ring_h[:, 2] = z_h

                draw_line(
                    sim,
                    ring_h,
                    rgba=(1.0, 0.0, 0.0, 0.5),
                    start_size=1.5,
                    end_size=1.5,
                )

        VIS_RANGE_GATE = 2.0

        for gi in range(self._n_gates):
            gp = self._gate_positions[gi]
            R = self._gate_rotmats[gi]
            h = self.GATE_HALF_OPENING

            is_active = gi >= self._target_gate
            gate_dist = float(np.linalg.norm(drone_pos - gp))

            blue = (0.2, 0.5, 1.0, 0.7) if is_active else (0.3, 0.3, 0.3, 0.3)

            opening_local = np.array([
                [0, -h, -h],
                [0, h, -h],
                [0, h, h],
                [0, -h, h],
                [0, -h, -h],
            ])

            draw_line(
                sim,
                (opening_local @ R.T) + gp,
                rgba=blue,
                start_size=2.0,
                end_size=2.0,
            )

            if gate_dist > VIS_RANGE_GATE:
                continue

            outer = self.GATE_OUTER_HALF + 0.08
            d = 0.15
            alpha = 0.55 if is_active else 0.2
            red = (1.0, 0.2, 0.2, alpha)

            for x_off in [-d, d]:
                face_local = np.array([
                    [x_off, -outer, -outer],
                    [x_off, outer, -outer],
                    [x_off, outer, outer],
                    [x_off, -outer, outer],
                    [x_off, -outer, -outer],
                ])

                draw_line(
                    sim,
                    (face_local @ R.T) + gp,
                    rgba=red,
                    start_size=2.0,
                    end_size=2.0,
                )

            for y_s, z_s in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                edge = np.array([
                    [-d, y_s * outer, z_s * outer],
                    [d, y_s * outer, z_s * outer],
                ])

                draw_line(
                    sim,
                    (edge @ R.T) + gp,
                    rgba=red,
                    start_size=2.0,
                    end_size=2.0,
                )

            if is_active:
                fl = self.FUNNEL_LENGTH
                fo = self.FUNNEL_OUTER_HALF

                for x_off in [-fl, -fl * 0.5, 0.0, fl * 0.5, fl]:
                    frac = abs(x_off) / fl if fl > 0 else 0.0
                    hw = h + frac * (fo - h)

                    funnel_rect = np.array([
                        [x_off, -hw, -hw],
                        [x_off, hw, -hw],
                        [x_off, hw, hw],
                        [x_off, -hw, hw],
                        [x_off, -hw, -hw],
                    ])

                    a_vis = 0.4 if abs(x_off) < 0.01 else 0.2

                    draw_line(
                        sim,
                        (funnel_rect @ R.T) + gp,
                        rgba=(0.2, 0.7, 1.0, a_vis),
                        start_size=1.5,
                        end_size=1.5,
                    )

                for y_s, z_s in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    edge_pts = []

                    for x_off in [-fl, 0.0, fl]:
                        frac = abs(x_off) / fl if fl > 0 else 0.0
                        hw = h + frac * (fo - h)
                        edge_pts.append([x_off, y_s * hw, z_s * hw])

                    draw_line(
                        sim,
                        (np.array(edge_pts) @ R.T) + gp,
                        rgba=(0.2, 0.7, 1.0, 0.25),
                        start_size=1.5,
                        end_size=1.5,
                    )