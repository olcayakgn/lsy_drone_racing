"""Gate-target MPC drone racing controller.

Architecture:
  The MPC IS the planner. No trajectory planner needed.
  - Reference: gate center (current + next gate for look-ahead)
  - Velocity reference: direction toward gate, scaled by desired speed
  - MPC optimizes the path considering drone dynamics + obstacle/gate constraints
  - Linearly interpolate reference from current position to gate for smooth guidance

  State: [pos(3), vel(3), u_prev(3)] = 9
  Control: [ax, ay, az] = 3
  Soft constraints: 4 obstacle dist² + 2 gate local y/z

Requires:
  - acados (ACADOS_SOURCE_DIR / ACADOS_INSTALL_DIR env vars)
  - MinGW-w64 in PATH (Windows C code compilation)
"""

from __future__ import annotations

import math
import os
import sys
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
_N_GATE_POST_SLOTS = 4  # gate frame corner avoidance slots
_TOTAL_CYL = _N_OBS_SLOTS + _N_GATE_POST_SLOTS
_NH = _TOTAL_CYL + 2  # cylindrical obstacles + gate opening y/z
_N_PARAMS = _TOTAL_CYL * 2 + 3 + 9  # obs_xy(16) + gate_pos(3) + gate_R(9) = 28


def _create_mpc_model() -> AcadosModel:
    """Double-integrator with parametric obstacle/gate constraints."""
    px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
    vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
    ax_prev, ay_prev, az_prev = cs.MX.sym("ax_prev"), cs.MX.sym("ay_prev"), cs.MX.sym("az_prev")

    x = cs.vertcat(px, py, pz, vx, vy, vz, ax_prev, ay_prev, az_prev)
    ax, ay, az = cs.MX.sym("ax"), cs.MX.sym("ay"), cs.MX.sym("az")
    u = cs.vertcat(ax, ay, az)
    p = cs.MX.sym("p", _N_PARAMS)

    g = 9.81
    tau = 0.02

    x_dot = cs.vertcat(
        vx, vy, vz,
        ax, ay, az - g,
        (ax - ax_prev) / tau,
        (ay - ay_prev) / tau,
        (az - az_prev) / tau,
    )

    h_list = []
    for i in range(_TOTAL_CYL):
        ox, oy = p[2 * i], p[2 * i + 1]
        h_list.append((px - ox) ** 2 + (py - oy) ** 2)

    gi = _TOTAL_CYL * 2
    gx, gy, gz = p[gi], p[gi + 1], p[gi + 2]
    ri = gi + 3
    dx, dy, dz = px - gx, py - gy, pz - gz
    p_local_y = p[ri + 1] * dx + p[ri + 4] * dy + p[ri + 7] * dz
    p_local_z = p[ri + 2] * dx + p[ri + 5] * dy + p[ri + 8] * dz
    h_list.append(p_local_y)
    h_list.append(p_local_z)

    con_h_expr = cs.vertcat(*h_list)

    model = AcadosModel()
    model.name = "gate_mpc_v2"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = x_dot
    model.con_h_expr = con_h_expr
    model.con_h_expr_e = con_h_expr
    return model


def _create_mpc_solver(dt: float, N: int, mass: float,
                       thrust_min: float, thrust_max: float,
                       obstacle_radius: float = 0.25,
                       gate_half_opening: float = 0.16,
                       gate_post_radius: float = 0.12):
    ocp = AcadosOcp()
    ocp.model = _create_mpc_model()

    nx, nu = 9, 3
    ny = nx + nu
    nh = _NH

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Position weights high — the MPC IS the planner, must reach gate
    Q_base = np.diag([
        80.0,  80.0,  80.0,   # position (drives toward gate — uniform)
        50.0,  50.0,  50.0,   # velocity (strong — enforces speed limit!)
        0.1,   0.1,   0.1,    # u_prev smoothing
    ])
    R_base = np.diag([0.15, 0.15, 0.15])  # input cost — penalize aggressive accel
    Q_e = np.diag([
        120.0, 120.0, 120.0,  # terminal position
        60.0,  60.0,  60.0,   # terminal velocity (must be at desired speed)
        0.05,  0.05,  0.05,
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

    a_min_z = thrust_min / mass
    a_max_z = thrust_max / mass
    a_max_xy = a_max_z * 0.7
    ocp.constraints.lbu = np.array([-a_max_xy, -a_max_xy, a_min_z])
    ocp.constraints.ubu = np.array([a_max_xy, a_max_xy, a_max_z])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.x0 = np.zeros(nx)

    r_sq = obstacle_radius ** 2
    r_sq_post = gate_post_radius ** 2
    lh = np.zeros(nh)
    uh = np.zeros(nh)
    lh[:_N_OBS_SLOTS] = r_sq
    uh[:_N_OBS_SLOTS] = 1e6
    lh[_N_OBS_SLOTS:_TOTAL_CYL] = r_sq_post
    uh[_N_OBS_SLOTS:_TOTAL_CYL] = 1e6
    lh[_TOTAL_CYL:] = -1000.0
    uh[_TOTAL_CYL:] = 1000.0

    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.idxsh = np.arange(nh)

    zl = np.zeros(nh)
    zu = np.zeros(nh)
    zl[:_N_OBS_SLOTS] = 300.0         # pole obstacle L1
    zl[_N_OBS_SLOTS:_TOTAL_CYL] = 150.0  # gate post L1
    zl[_TOTAL_CYL:] = 100.0          # gate opening L1
    zu[_TOTAL_CYL:] = 100.0          # gate opening upper L1
    ocp.cost.zl = zl
    ocp.cost.zu = zu
    Zl = np.zeros(nh)
    Zu = np.zeros(nh)
    Zl[:_N_OBS_SLOTS] = 1400.0        # pole obstacle L2
    Zl[_N_OBS_SLOTS:_TOTAL_CYL] = 800.0  # gate post L2
    Zl[_TOTAL_CYL:] = 400.0          # gate opening L2
    Zu[_TOTAL_CYL:] = 400.0          # gate opening upper L2
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
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.tf = dt * N

    code_dir = Path(__file__).parent / "c_generated_code"
    code_dir.mkdir(exist_ok=True)

    solver = AcadosOcpSolver(
        ocp,
        json_file=str(code_dir / "gate_mpc_v2.json"),
        verbose=False, build=True, generate=True,
    )

    # Non-uniform: weight near stages more
    gamma = 0.95
    weights = np.array([gamma ** i for i in range(N)])
    weights *= N / weights.sum()
    for i in range(N):
        solver.cost_set(i, "W", weights[i] * W_base)
    solver.cost_set(N, "W", Q_e)

    return solver, nx, nu, W_base, Q_e


def accel_to_attitude(a_cmd: np.ndarray, yaw_des: float, mass: float) -> np.ndarray:
    a_mag = float(np.linalg.norm(a_cmd))
    if a_mag < 0.5:
        return np.array([0.0, 0.0, yaw_des, mass * 9.81], dtype=np.float64)

    thrust = mass * a_mag
    z_des = a_cmd / a_mag
    cy, sy = np.cos(yaw_des), np.sin(yaw_des)
    zx_local = cy * z_des[0] + sy * z_des[1]
    zy_local = -sy * z_des[0] + cy * z_des[1]
    zz_local = z_des[2]

    roll = float(np.arctan2(-zy_local, zz_local))
    pitch = float(np.arctan2(zx_local, np.sqrt(zy_local**2 + zz_local**2)))
    roll = np.clip(roll, -0.6, 0.6)
    pitch = np.clip(pitch, -0.6, 0.6)
    return np.array([roll, pitch, yaw_des, thrust], dtype=np.float64)


class PMMRacingController(Controller):
    """Gate-target MPC controller for drone racing.

    The MPC IS the planner:
      1. Set reference = interpolate from drone position toward gate center
      2. Velocity reference = direction toward gate, scaled by desired speed
      3. MPC optimizes under obstacle/gate soft constraints
      4. Convert accel → attitude command
    """

    OBSTACLE_RADIUS = 0.25
    GATE_HALF_OPENING = 0.16
    GATE_OUTER_HALF = 0.28       # center of gate frame bars (gate-local coords)
    GATE_POST_RADIUS = 0.12      # avoidance radius for gate frame bars
    FUNNEL_LENGTH = 1.0          # funnel starts narrowing this far from gate plane (m)
    FUNNEL_OUTER_HALF = 0.50     # funnel width far from gate (m) — wide entry
    APPROACH_DIST = 0.30
    EXIT_DIST = 0.30
    GROUND_CLEARANCE = 0.10
    CEILING = 1.80
    V_CRUISE = 1.5          # desired cruise speed toward gate
    V_GATE = 1.0            # desired speed at gate passage
    MPC_N = 50               # longer horizon for planning
    MPC_DT = 0.05            # 2.0s horizon
    APF_INFLUENCE = 0.35
    APF_GAIN = 0.30
    APF_MAX = 0.60

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

        self._obstacle_positions = np.array(
            [g.tolist() for g in obs["obstacles_pos"]], dtype=np.float64)
        self._gates_visited = obs["gates_visited"].copy()
        self._obstacles_visited = np.array(
            obs.get("obstacles_visited", np.zeros(len(self._obstacle_positions), dtype=bool)),
            dtype=bool)

        print("[GATE-MPC] Building MPC solver...")
        self._solver, self._nx, self._nu, self._W_base, self._Q_e = _create_mpc_solver(
            dt=self.MPC_DT, N=self.MPC_N, mass=self._mass,
            thrust_min=self._thrust_min, thrust_max=self._thrust_max,
            obstacle_radius=self.OBSTACLE_RADIUS,
            gate_half_opening=self.GATE_HALF_OPENING,
            gate_post_radius=self.GATE_POST_RADIUS,
        )
        self._nh = _NH
        self._n_params = _N_PARAMS
        print(f"[GATE-MPC] Solver ready. N={self.MPC_N}, dt={self.MPC_DT:.3f}s, "
              f"horizon={self.MPC_N * self.MPC_DT:.1f}s")

        self._prev_accel = np.array([0.0, 0.0, self._g])
        self._prev_output = np.array([0.0, 0.0, 0.0, self._mass * self._g])
        self._tick = 0
        self._finished = False
        self._mass_estimate = self._mass

    def _get_gate_normal(self, gi: int, from_pos: np.ndarray) -> np.ndarray:
        """Get gate normal pointing from from_pos toward gate."""
        normal = self._gate_rotmats[gi][:, 0].copy()
        to_gate = self._gate_positions[gi] - from_pos
        if np.dot(to_gate, normal) < 0:
            normal = -normal
        return normal

    def _generate_references(self, pos: np.ndarray, vel: np.ndarray):
        """Generate MPC reference positions and velocities.

        Strategy: linear interpolation from current pos to gate center,
        then from gate exit to next gate. Velocity ref = direction × desired speed.
        """
        ref_pos = np.zeros((self.MPC_N + 1, 3))
        ref_vel = np.zeros((self.MPC_N + 1, 3))

        if self._target_gate < 0 or self._target_gate >= self._n_gates:
            ref_pos[:] = pos
            return ref_pos, ref_vel

        gp = self._gate_positions[self._target_gate]
        normal = self._get_gate_normal(self._target_gate, pos)
        approach_pt = gp - self.APPROACH_DIST * normal
        exit_pt = gp + self.EXIT_DIST * normal

        # Next gate info for look-ahead
        next_gate = self._target_gate + 1
        has_next = next_gate < self._n_gates
        if has_next:
            next_gp = self._gate_positions[next_gate]
        else:
            next_gp = exit_pt  # just coast past exit

        # Build waypoint sequence, skip approach if already close to gate
        dist_to_gate = float(np.linalg.norm(pos - gp))
        if dist_to_gate > self.APPROACH_DIST * 1.5:
            waypoints = [pos.copy(), approach_pt, gp.copy(), exit_pt]
        else:
            # Already close — go straight through gate, don't backtrack to approach
            waypoints = [pos.copy(), gp.copy(), exit_pt]
        if has_next:
            next_normal = self._get_gate_normal(next_gate, exit_pt)
            next_approach = next_gp - self.APPROACH_DIST * next_normal
            waypoints.append(next_approach)
            waypoints.append(next_gp.copy())

        # Compute cumulative distances along waypoints
        cum_dist = [0.0]
        for i in range(1, len(waypoints)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(
                np.array(waypoints[i]) - np.array(waypoints[i-1])))
        total_dist = cum_dist[-1]

        if total_dist < _EPS:
            ref_pos[:] = pos
            return ref_pos, ref_vel

        # Estimate time to reach gate based on current speed + cruise speed
        speed = max(float(np.linalg.norm(vel)), 0.3)
        dist_to_gate = float(np.linalg.norm(gp - pos))

        # For each horizon stage, compute how far along the path we'd be
        # assuming we travel at V_CRUISE, starting from current speed
        for i in range(self.MPC_N + 1):
            t = i * self.MPC_DT
            # Distance traveled: average of current speed and cruise
            avg_speed = min(self.V_CRUISE, (speed + self.V_CRUISE) / 2.0)
            s = avg_speed * t  # arc-length along path

            # Clamp to total path length
            s = min(s, total_dist)

            # Interpolate position along waypoint chain
            for j in range(len(cum_dist) - 1):
                if s <= cum_dist[j + 1] or j == len(cum_dist) - 2:
                    seg_len = cum_dist[j + 1] - cum_dist[j]
                    if seg_len > _EPS:
                        alpha = (s - cum_dist[j]) / seg_len
                    else:
                        alpha = 0.0
                    alpha = max(0.0, min(1.0, alpha))
                    ref_pos[i] = (1 - alpha) * np.array(waypoints[j]) + \
                                 alpha * np.array(waypoints[j + 1])
                    break

            # Clamp altitude
            ref_pos[i, 2] = np.clip(ref_pos[i, 2], self.GROUND_CLEARANCE, self.CEILING)

            # Velocity reference: always fly THROUGH the gate at cruise speed
            if i < self.MPC_N:
                dist_to_g = float(np.linalg.norm(ref_pos[i] - gp))
                if dist_to_g < 0.6:
                    # Near gate: velocity along gate normal (fly THROUGH, don't slow down)
                    ref_vel[i] = normal * self.V_CRUISE
                else:
                    direction = gp - ref_pos[i]
                    d_norm = float(np.linalg.norm(direction))
                    if d_norm > _EPS:
                        ref_vel[i] = direction / d_norm * self.V_CRUISE
                    else:
                        ref_vel[i] = np.zeros(3)

                # Cap vertical velocity component to prevent runaway ascent/descent
                # The drone should prioritize horizontal progress, not climb too fast
                max_vz = self.V_CRUISE * 0.5  # vertical speed capped at 50% of cruise
                ref_vel[i, 2] = np.clip(ref_vel[i, 2], -max_vz, max_vz)

        # Terminal velocity: toward next gate or zero
        if has_next:
            direction = next_gp - ref_pos[self.MPC_N]
            d_norm = float(np.linalg.norm(direction))
            if d_norm > _EPS:
                ref_vel[self.MPC_N] = direction / d_norm * self.V_CRUISE
                ref_vel[self.MPC_N, 2] = np.clip(
                    ref_vel[self.MPC_N, 2], -self.V_CRUISE * 0.5, self.V_CRUISE * 0.5)
        else:
            ref_vel[self.MPC_N] = np.zeros(3)

        return ref_pos, ref_vel

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None,
    ) -> NDArray[np.floating]:
        self._tick += 1
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        self._last_pos = pos.copy()

        if self._finished:
            return np.array([0.0, 0.0, 0.0, self._mass * self._g], dtype=np.float32)

        # Generate references: interpolated path to gate
        ref_pos, ref_vel = self._generate_references(pos, vel)

        # MPC initial state
        x0 = np.concatenate([pos, vel, self._prev_accel])
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # Constraint parameters
        params = np.zeros(self._n_params)
        n_obs = min(len(self._obstacle_positions), _N_OBS_SLOTS)
        for oi in range(n_obs):
            params[2 * oi] = self._obstacle_positions[oi][0]
            params[2 * oi + 1] = self._obstacle_positions[oi][1]
        for oi in range(n_obs, _N_OBS_SLOTS):
            params[2 * oi] = 100.0
            params[2 * oi + 1] = 100.0

        # Gate frame corner obstacles (4 corners of target gate frame)
        if 0 <= self._target_gate < self._n_gates:
            gp_frame = self._gate_positions[self._target_gate]
            R_frame = self._gate_rotmats[self._target_gate]
            oh = self.GATE_OUTER_HALF
            corners_local = np.array([
                [0, -oh, -oh], [0, oh, -oh],
                [0, -oh, oh],  [0, oh, oh],
            ])
            for ci in range(4):
                corner_world = gp_frame + R_frame @ corners_local[ci]
                pi = _N_OBS_SLOTS + ci
                params[2 * pi] = corner_world[0]
                params[2 * pi + 1] = corner_world[1]
        else:
            for ci in range(_N_GATE_POST_SLOTS):
                pi = _N_OBS_SLOTS + ci
                params[2 * pi] = 100.0
                params[2 * pi + 1] = 100.0

        gi_p = _TOTAL_CYL * 2
        if 0 <= self._target_gate < self._n_gates:
            gp_cur = self._gate_positions[self._target_gate]
            R_cur = self._gate_rotmats[self._target_gate]
        else:
            gp_cur = np.array([100.0, 100.0, 100.0])
            R_cur = np.eye(3)

        # Next gate info for corridor
        next_gate_idx = self._target_gate + 1
        has_next = 0 <= next_gate_idx < self._n_gates
        if has_next:
            gp_next = self._gate_positions[next_gate_idx]
            R_next = self._gate_rotmats[next_gate_idx]
        else:
            gp_next = None
            R_next = None

        r_sq = self.OBSTACLE_RADIUS ** 2
        r_sq_post = self.GATE_POST_RADIUS ** 2
        h_open = self.GATE_HALF_OPENING

        # Per-stage params and constraints: continuous corridor
        # For each stage, pick whichever gate the stage is closer to (in the
        # flight direction). Stages past the current gate use the next gate.
        normal_cur = self._get_gate_normal(self._target_gate, pos) \
            if 0 <= self._target_gate < self._n_gates else np.array([1, 0, 0])

        for i in range(self.MPC_N + 1):
            # Decide which gate this stage should be constrained by
            use_next = False
            if 0 <= self._target_gate < self._n_gates:
                p_local_cur = R_cur.T @ (ref_pos[i] - gp_cur)
                # Past current gate exit and next gate exists?
                if p_local_cur[0] > self.EXIT_DIST and has_next:
                    use_next = True

            # Set parameters for this stage (gate position + rotation)
            params_i = params.copy()
            if use_next:
                # Gate frame posts for next gate
                oh = self.GATE_OUTER_HALF
                corners_local = np.array([
                    [0, -oh, -oh], [0, oh, -oh],
                    [0, -oh, oh],  [0, oh, oh],
                ])
                for ci in range(4):
                    corner_world = gp_next + R_next @ corners_local[ci]
                    pi = _N_OBS_SLOTS + ci
                    params_i[2 * pi] = corner_world[0]
                    params_i[2 * pi + 1] = corner_world[1]
                # Gate opening constraint relative to next gate
                params_i[gi_p:gi_p + 3] = gp_next
                params_i[gi_p + 3:gi_p + 12] = R_next.flatten()
            else:
                params_i[gi_p:gi_p + 3] = gp_cur
                params_i[gi_p + 3:gi_p + 12] = R_cur.flatten()

            self._solver.set(i, "p", params_i)

        # Constraint bounds per stage
        for i in range(1, self.MPC_N + 1):
            lh_i = np.zeros(self._nh)
            uh_i = np.zeros(self._nh)
            lh_i[:_N_OBS_SLOTS] = r_sq
            uh_i[:_N_OBS_SLOTS] = 1e6
            lh_i[_N_OBS_SLOTS:_TOTAL_CYL] = r_sq_post
            uh_i[_N_OBS_SLOTS:_TOTAL_CYL] = 1e6

            if 0 <= self._target_gate < self._n_gates:
                # Determine which gate this stage is constrained by
                p_local_cur = R_cur.T @ (ref_pos[i] - gp_cur)
                use_next = p_local_cur[0] > self.EXIT_DIST and has_next

                if use_next:
                    # Funnel toward NEXT gate
                    p_local_next = R_next.T @ (ref_pos[i] - gp_next)
                    dist_to_plane = abs(p_local_next[0])
                    if dist_to_plane < self.FUNNEL_LENGTH:
                        alpha = dist_to_plane / self.FUNNEL_LENGTH
                        h_bound = h_open + alpha * (self.FUNNEL_OUTER_HALF - h_open)
                        lh_i[_TOTAL_CYL] = -h_bound
                        uh_i[_TOTAL_CYL] = h_bound
                        lh_i[_TOTAL_CYL + 1] = -h_bound
                        uh_i[_TOTAL_CYL + 1] = h_bound
                    else:
                        # Transition zone: use wide corridor to connect
                        lh_i[_TOTAL_CYL] = -self.FUNNEL_OUTER_HALF
                        uh_i[_TOTAL_CYL] = self.FUNNEL_OUTER_HALF
                        lh_i[_TOTAL_CYL + 1] = -self.FUNNEL_OUTER_HALF
                        uh_i[_TOTAL_CYL + 1] = self.FUNNEL_OUTER_HALF
                else:
                    # Funnel toward CURRENT gate
                    dist_to_plane = abs(p_local_cur[0])
                    if dist_to_plane < self.FUNNEL_LENGTH:
                        alpha = dist_to_plane / self.FUNNEL_LENGTH
                        h_bound = h_open + alpha * (self.FUNNEL_OUTER_HALF - h_open)
                        lh_i[_TOTAL_CYL] = -h_bound
                        uh_i[_TOTAL_CYL] = h_bound
                        lh_i[_TOTAL_CYL + 1] = -h_bound
                        uh_i[_TOTAL_CYL + 1] = h_bound
                    else:
                        lh_i[_TOTAL_CYL:] = -1000.0
                        uh_i[_TOTAL_CYL:] = 1000.0
            else:
                lh_i[_TOTAL_CYL:] = -1000.0
                uh_i[_TOTAL_CYL:] = 1000.0

            if i < self.MPC_N:
                self._solver.constraints_set(i, "lh", lh_i)
                self._solver.constraints_set(i, "uh", uh_i)
            else:
                # Terminal
                self._solver.constraints_set(self.MPC_N, "lh", lh_i)
                self._solver.constraints_set(self.MPC_N, "uh", uh_i)

        # Set references + dynamic weights
        gamma = 0.95
        weights_raw = np.array([gamma ** k for k in range(self.MPC_N)])
        weights_raw *= self.MPC_N / weights_raw.sum()

        for i in range(self.MPC_N):
            yref = np.zeros(self._nx + self._nu)
            yref[0:3] = ref_pos[i]
            yref[3:6] = ref_vel[i]
            yref[6:9] = np.array([0.0, 0.0, self._g])  # hover accel as u_prev ref
            yref[9:12] = np.array([0.0, 0.0, self._g])  # hover accel as u ref
            self._solver.set(i, "yref", yref)

            # Near gate: boost VELOCITY weight to maintain momentum through
            W_i = weights_raw[i] * self._W_base
            if 0 <= self._target_gate < self._n_gates:
                gate_dist = float(np.linalg.norm(ref_pos[i] - gp_cur))
                if gate_dist < 0.6:
                    boost = 1.0 + 4.0 * (1.0 - gate_dist / 0.6)
                    W_i = W_i.copy()
                    # Boost velocity weights to maintain speed through gate
                    W_i[3, 3] *= boost
                    W_i[4, 4] *= boost
                    W_i[5, 5] *= boost
            self._solver.cost_set(i, "W", W_i)

        # Terminal reference
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = ref_pos[self.MPC_N]
        yref_e[3:6] = ref_vel[self.MPC_N]
        yref_e[6:9] = np.array([0.0, 0.0, self._g])
        self._solver.set(self.MPC_N, "yref", yref_e)

        status = self._solver.solve()

        if status <= 2:
            u0 = self._solver.get(0, "u")
        else:
            pull = ref_pos[3] - pos
            pull_dist = np.linalg.norm(pull)
            if pull_dist > 0.01:
                u0 = pull / pull_dist * min(4.0, pull_dist * 8.0)
                u0[2] += self._g
            else:
                u0 = np.array([0.0, 0.0, self._g])

        self._prev_accel = u0.copy()

        # Debug
        if self._tick % 5 == 0:
            speed = float(np.linalg.norm(vel))
            vz = float(vel[2])
            if 0 <= self._target_gate < self._n_gates:
                gate_dist = float(np.linalg.norm(pos - self._gate_positions[self._target_gate]))
                gp_t = self._gate_positions[self._target_gate]
                dz = float(gp_t[2] - pos[2])
            else:
                gate_dist = -1.0
                dz = 0.0
            print(f"[GATE-MPC] step={self._tick} pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] "
                  f"v={speed:.2f} vz={vz:.2f} dz={dz:.2f} gate_dist={gate_dist:.3f} gate={self._target_gate}")

        # Accel → attitude
        current_rpy = Rot.from_quat(obs["quat"]).as_euler("xyz")
        current_yaw = current_rpy[2]

        # Yaw toward gate
        if 0 <= self._target_gate < self._n_gates:
            to_gate = self._gate_positions[self._target_gate] - pos
            if np.linalg.norm(to_gate[:2]) > 0.1:
                yaw_des = float(np.arctan2(to_gate[1], to_gate[0]))
            else:
                yaw_des = current_yaw
        else:
            yaw_des = current_yaw

        yaw_error = yaw_des - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        yaw_cmd = current_yaw + np.clip(0.15 * yaw_error, -0.1, 0.1)

        output = accel_to_attitude(u0, yaw_cmd, self._mass_estimate)
        roll_cmd, pitch_cmd, yaw_out, thrust_cmd = output

        # APF reactive layer
        apf_force = np.zeros(3)
        for op in self._obstacle_positions:
            diff = pos[:2] - op[:2]
            d = float(np.linalg.norm(diff))
            if _EPS < d < self.APF_INFLUENCE:
                mag = min(
                    self.APF_GAIN * (1.0 / d - 1.0 / self.APF_INFLUENCE) / (d * d),
                    self.APF_MAX)
                apf_force[:2] += mag * diff / d

        if np.linalg.norm(apf_force) > 0.01:
            roll_cmd += float(np.clip(
                -apf_force[1] / (self._mass_estimate * self._g) * 0.3, -0.1, 0.1))
            pitch_cmd += float(np.clip(
                apf_force[0] / (self._mass_estimate * self._g) * 0.3, -0.1, 0.1))

        # Ground/ceiling safety
        z = pos[2]
        if z < self.GROUND_CLEARANCE + 0.1:
            thrust_cmd += 0.5 * (self.GROUND_CLEARANCE + 0.1 - z) * self._mass_estimate * self._g
        if z > self.CEILING - 0.2:
            # Strong ceiling avoidance — proportional to overshoot
            overshoot = z - (self.CEILING - 0.2)
            thrust_cmd -= (0.5 + 2.0 * overshoot) * overshoot * self._mass_estimate * self._g
            thrust_cmd = max(thrust_cmd, self._thrust_min)

        roll_cmd = float(np.clip(roll_cmd, -0.55, 0.55))
        pitch_cmd = float(np.clip(pitch_cmd, -0.55, 0.55))
        thrust_cmd = float(np.clip(thrust_cmd, self._thrust_min, self._thrust_max))

        self._prev_output = np.array([roll_cmd, pitch_cmd, yaw_out, thrust_cmd])
        return self._prev_output.astype(np.float32)

    def step_callback(
        self, action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float, terminated: bool, truncated: bool, info: dict,
    ) -> bool:
        new_target = int(obs["target_gate"])
        gate_changed = new_target != self._target_gate
        self._target_gate = new_target

        if new_target < 0:
            self._finished = True
            return True

        # Update gate/obstacle positions from sensor
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

        # Simple mass estimation
        if len(action) >= 4 and self._tick > 5:
            thrust = action[3]
            if abs(thrust) > 0.01:
                mass_obs = thrust / (self._g + 0.01)
                self._mass_estimate = 0.99 * self._mass_estimate + 0.01 * mass_obs

        return self._finished

    def render_callback(self, sim: Sim):
        from crazyflow.sim.visualize import draw_line, draw_points

        drone_pos = getattr(self, '_last_pos', np.zeros(3))

        # Draw gate targets
        if 0 <= self._target_gate < self._n_gates:
            gp = self._gate_positions[self._target_gate]
            draw_points(sim, gp.reshape(1, -1), rgba=(1.0, 1.0, 0.0, 1.0), size=0.04)

            # Draw line from drone to gate
            draw_line(sim, np.vstack([drone_pos, gp]),
                      rgba=(0.0, 1.0, 0.0, 0.6), start_size=2.0, end_size=2.0)

        # Obstacle zones
        VIS_RANGE = 1.5
        for op in self._obstacle_positions:
            if np.linalg.norm(drone_pos[:2] - op[:2]) > VIS_RANGE:
                continue
            n_ring = 24
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            ring_pts = np.zeros((n_ring, 3))
            for j in range(n_ring):
                ring_pts[j, 0] = op[0] + self.OBSTACLE_RADIUS * np.cos(angles[j])
                ring_pts[j, 1] = op[1] + self.OBSTACLE_RADIUS * np.sin(angles[j])
                ring_pts[j, 2] = 0.5
            ring_closed = np.vstack([ring_pts, ring_pts[0:1]])
            draw_line(sim, ring_closed, rgba=(1.0, 0.0, 0.0, 0.7),
                      start_size=2.0, end_size=2.0)
            for z_h in [0.2, 0.8, 1.2]:
                ring_h = ring_closed.copy()
                ring_h[:, 2] = z_h
                draw_line(sim, ring_h, rgba=(1.0, 0.0, 0.0, 0.5),
                          start_size=1.5, end_size=1.5)

        # Gate openings + forbidden slab
        VIS_RANGE_GATE = 2.0
        for gi in range(self._n_gates):
            gp = self._gate_positions[gi]
            R = self._gate_rotmats[gi]
            h = self.GATE_HALF_OPENING
            is_active = gi >= self._target_gate
            gate_dist = float(np.linalg.norm(drone_pos - gp))

            blue = (0.2, 0.5, 1.0, 0.7) if is_active else (0.3, 0.3, 0.3, 0.3)
            opening_local = np.array([
                [0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h], [0, -h, -h]])
            draw_line(sim, (opening_local @ R.T) + gp, rgba=blue,
                      start_size=2.0, end_size=2.0)

            # Forbidden slab — only when gate is in vision
            if gate_dist > VIS_RANGE_GATE:
                continue

            outer = self.GATE_OUTER_HALF + 0.08  # extend to outer edge of frame
            d = 0.15  # depth of forbidden slab along gate normal
            alpha = 0.55 if is_active else 0.2
            red = (1.0, 0.2, 0.2, alpha)

            # Front and back faces of the forbidden slab
            for x_off in [-d, d]:
                face_local = np.array([
                    [x_off, -outer, -outer], [x_off, outer, -outer],
                    [x_off, outer, outer], [x_off, -outer, outer],
                    [x_off, -outer, -outer],
                ])
                draw_line(sim, (face_local @ R.T) + gp, rgba=red,
                          start_size=2.0, end_size=2.0)

            # 4 depth edges connecting front/back outer corners
            for y_s, z_s in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                edge = np.array([[-d, y_s * outer, z_s * outer],
                                 [d, y_s * outer, z_s * outer]])
                draw_line(sim, (edge @ R.T) + gp, rgba=red,
                          start_size=2.0, end_size=2.0)

            # Funnel visualization (MPCC++ tunnel) — translucent blue
            if is_active:
                fl = self.FUNNEL_LENGTH
                fo = self.FUNNEL_OUTER_HALF
                # Draw funnel cross-sections at a few depths along gate normal
                for x_off in [-fl, -fl * 0.5, 0.0, fl * 0.5, fl]:
                    frac = abs(x_off) / fl if fl > 0 else 0.0
                    hw = h + frac * (fo - h)  # interpolated half-width
                    funnel_rect = np.array([
                        [x_off, -hw, -hw], [x_off, hw, -hw],
                        [x_off, hw, hw], [x_off, -hw, hw],
                        [x_off, -hw, -hw],
                    ])
                    a_vis = 0.4 if abs(x_off) < 0.01 else 0.2
                    draw_line(sim, (funnel_rect @ R.T) + gp,
                              rgba=(0.2, 0.7, 1.0, a_vis),
                              start_size=1.5, end_size=1.5)
                # Connect funnel corners along depth (4 edges)
                for y_s, z_s in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    edge_pts = []
                    for x_off in [-fl, 0.0, fl]:
                        frac = abs(x_off) / fl if fl > 0 else 0.0
                        hw = h + frac * (fo - h)
                        edge_pts.append([x_off, y_s * hw, z_s * hw])
                    draw_line(sim, (np.array(edge_pts) @ R.T) + gp,
                              rgba=(0.2, 0.7, 1.0, 0.25),
                              start_size=1.5, end_size=1.5)
