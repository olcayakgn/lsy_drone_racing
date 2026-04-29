"""SFC based hybrid adaptive controller.

S. Liu et al., "Planning Dynamically Feasible Trajectories for Quadrotors Using Safe
Flight Corridors in 3-D Complex Environment".         
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cvxpy as cp
import numpy as np
from crazyflow.sim.visualize import draw_line
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class WayNode(NamedTuple):
    """A waypoint along the planned route, with optional gate-frame data."""

    pos: NDArray
    is_gate: bool
    normal: NDArray | None
    right: NDArray | None
    up: NDArray | None
    is_authored: bool = False


class GateTransition(NamedTuple):
    """Hardcoded auxiliary waypoints surrounding one gate - gate3."""

    entry_swing_lat: float
    exit_detour: tuple[tuple[float, float, float], ...]
    next_pre_override: tuple[float, float, float] | None


# Per-gate transition template for the level2 track (4 gates, fixed order).
LEVEL2_TRANSITIONS: tuple[GateTransition, ...] = (
    GateTransition(0.0, ((0.45, 0.0, 0.0), (1.45, 0.0, 0.0), (0.75, 1.0, 0.0)), None),
    # Gate 1: straight pass-through, default post-anchor.
    GateTransition(0.0, (), None),
    # Gate 3 (1-indexed): hairpin out toward gate 4. The climb from gate-3
    # altitude (z=0.70) up to gate-4 altitude (z=1.20) is deferred to the
    # WP1->WP2 leg — that leg is far west (x<=-1.45) so the climb through
    # the top bar's z range happens with ample xy clearance from the bar.
    # 4-waypoint detour:
    #   WP0: clean exit through opening (z=0.70)
    #   WP1: SW at gate-3 altitude. Forward extended to 0.60 so the WP0->WP1
    #        chord clears the right-bar capsule by 0.032 m (perpendicular
    #        distance 0.292 m vs 0.260 m capsule radius). Pole 3's worst-case
    #        randomized position can sit on this segment — handled by the
    #        deflection re-enabled for poles on authored segments.
    #   WP2: SW + climb to z=1.35 (above right-bar top capsule cap at z=1.32)
    #   WP3: back east at gate-4 altitude
    # Pre-anchor for gate 4 lifted to dz=0.10 so the WP3-pre4 chord midpoint
    # also clears the bar's top hemisphere.
    GateTransition(
        0.0,
        ((0.12, 0.0, 0.0), (0.60, 0.40, 0.0), (0.45, 0.50, 0.65), (0.10, 0.55, 0.55)),
        (-0.35, 0.30, 0.10),
    ),
    # Gate 3: terminal pass-through.
    GateTransition(0.0, (), None),
)


class CapsuleObs(NamedTuple):
    """Capsule-shaped obstacle (cylinder body + spherical end caps)."""

    a: NDArray
    b: NDArray
    radius: float
    is_gate: bool


class ConvexHull:
    """Convex polytope expressed as an intersection of half-spaces A x <= b."""

    def __init__(self, a: NDArray, b: NDArray) -> None:
        """Initialize the corridor between two skeleton anchors.

        Args:
            a: Segment start point.
            b: Segment end point.
        """
        self.A: list[NDArray] = []
        self.b: list[float] = []
        self.a = a
        self.b_pt = b

        # Arena bounding box
        self.push_plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 3.0]))
        self.push_plane(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -0.2]))
        self.push_plane(np.array([1.0, 0.0, 0.0]), np.array([15.0, 0.0, 0.0]))
        self.push_plane(np.array([-1.0, 0.0, 0.0]), np.array([-15.0, 0.0, 0.0]))
        self.push_plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 15.0, 0.0]))
        self.push_plane(np.array([0.0, -1.0, 0.0]), np.array([0.0, -15.0, 0.0]))

    def push_plane(self, n: NDArray, p: NDArray) -> None:
        """Append the constraint n . (x - p) <= 0 (n is the outward normal)."""
        self.A.append(n)
        self.b.append(float(np.dot(n, p)))


def segment_segment_closest(
    a0: NDArray, a1: NDArray, b0: NDArray, b1: NDArray
) -> tuple[NDArray, NDArray]:
    """Closest points on segment a0-a1 and segment b0-b1 (Ericson, RTCD 5.1.9)."""
    da = a1 - a0
    db = b1 - b0
    r = a0 - b0
    aa = np.dot(da, da)
    ee = np.dot(db, db)
    f = np.dot(db, r)

    if aa <= 1e-6 and ee <= 1e-6:
        return a0, b0
    if aa <= 1e-6:
        v = np.clip(f / ee, 0.0, 1.0)
        return a0, b0 + v * db

    c = np.dot(da, r)
    if ee <= 1e-6:
        u = np.clip(-c / aa, 0.0, 1.0)
        return a0 + u * da, b0

    bb = np.dot(da, db)
    denom = aa * ee - bb * bb

    if denom != 0.0:
        u = np.clip((bb * f - c * ee) / denom, 0.0, 1.0)
    else:
        u = 0.0

    v = (bb * u + f) / ee
    if v < 0.0:
        v = 0.0
        u = np.clip(-c / aa, 0.0, 1.0)
    elif v > 1.0:
        v = 1.0
        u = np.clip((bb - c) / aa, 0.0, 1.0)

    return a0 + u * da, b0 + v * db


class CorridorController(Controller):
    """B-spline trajectory controller with convex-corridor obstacle constraints."""

    K_SMOOTH_VEL = 1.5
    K_SMOOTH_ACC = 7.0
    K_SMOOTH_JERK = 12.0
    K_CENTERLINE = 0.02

    K_START_POS = 8.0
    K_START_VEL = 60.0
    K_GATE_AXIS = 80.0

    LOOKAHEAD_DT = 0.04

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:
        """Initialize the corridor-based trajectory controller.

        Args:
            obs: Initial observation dictionary.
            info: Environment info dictionary.
            config: Configuration dictionary.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._spline_tick = 0
        self._finished = False

        self.anchor_gap = 0.45
        self.cruise_speed = 1.0
        self.ctrl_pts_per_seg = 4

        self.gate_outer_w = 0.72
        self.gate_inner_w = 0.40
        self.gate_frame_t = (self.gate_outer_w - self.gate_inner_w) / 2.0
        self.gate_thickness = 0.10

        self.pole_radius = 0.015
        self.pole_height = 1.52

        self.clearance = 0.18

        self.gates_pos = obs["gates_pos"].copy()
        self.gates_quat = obs["gates_quat"].copy()
        self.obstacles_pos = obs.get("obstacles_pos", np.array([]))
        self.target_gate_idx = 0
        self._prev_pos: NDArray | None = None

        v0 = obs.get("vel", np.zeros(3))
        self._build_trajectory(obs["pos"], v0)

    # ------------------------------------------------------------------ planner
    def _build_trajectory(self, pos_now: NDArray, vel_now: NDArray) -> None:
        """Build the B-spline reference for the remaining gates.

        Args:
            pos_now: Current drone position.
            vel_now: Current drone velocity.
        """
        skeleton = self._build_skeleton(pos_now[:3])
        self._anchor_pos = pos_now[:3].copy()

        capsules = self._collect_capsules()
        corridors = self._build_corridors(skeleton, capsules)

        ctrl_pts = self._solve_qp(skeleton, corridors, vel_now)

        order = 3
        n = len(ctrl_pts)

        chord = np.maximum(np.linalg.norm(np.diff(ctrl_pts, axis=0), axis=1), 1e-4)
        u = np.concatenate(([0.0], np.cumsum(chord)))
        if u[-1] > 0:
            u /= u[-1]

        knots = np.zeros(n + order + 1)
        knots[: order + 1] = 0.0
        knots[-order - 1 :] = 1.0
        for i in range(1, n - order):
            knots[i + order] = np.mean(u[i : i + order])

        self._des_pos_spline = BSpline(knots, ctrl_pts, order)
        self._t_total = float(np.sum(chord) / self.cruise_speed)
        self._spline_tick = 0

    def _collect_capsules(self) -> list[CapsuleObs]:
        """Wrap each pole and gate-frame bar/stand as a 3D capsule obstacle."""
        out: list[CapsuleObs] = []
        m = self.clearance

        # Vertical poles
        for p in self.obstacles_pos:
            out.append(
                CapsuleObs(
                    np.array([p[0], p[1], 0.0]),
                    np.array([p[0], p[1], self.pole_height]),
                    self.pole_radius + m,
                    False,
                )
            )

        # Gates: stand below + four frame bars
        for pos, quat in zip(self.gates_pos, self.gates_quat):
            rot = R.from_quat(quat)
            up = rot.apply([0.0, 0.0, 1.0])
            right = rot.apply([0.0, 1.0, 0.0])

            stand_h = pos[2] - self.gate_outer_w / 2.0
            if stand_h > 0:
                out.append(
                    CapsuleObs(
                        pos - up * (self.gate_outer_w / 2.0),
                        pos - up * (self.gate_outer_w / 2.0 + stand_h),
                        0.05 + m,
                        True,
                    )
                )

            bar_dist = 0.28
            bar_r = 0.08 + m

            out.append(
                CapsuleObs(
                    pos + up * bar_dist - right * 0.36,
                    pos + up * bar_dist + right * 0.36,
                    bar_r,
                    True,
                )
            )
            out.append(
                CapsuleObs(
                    pos - up * bar_dist - right * 0.36,
                    pos - up * bar_dist + right * 0.36,
                    bar_r,
                    True,
                )
            )
            out.append(
                CapsuleObs(
                    pos - right * bar_dist + up * 0.36,
                    pos - right * bar_dist - up * 0.36,
                    bar_r,
                    True,
                )
            )
            out.append(
                CapsuleObs(
                    pos + right * bar_dist + up * 0.36,
                    pos + right * bar_dist - up * 0.36,
                    bar_r,
                    True,
                )
            )

        return out

    def _build_corridors(
        self, skeleton: list[WayNode], capsules: list[CapsuleObs]
    ) -> list[ConvexHull]:
        """Build a convex polytope per skeleton segment using separating planes."""
        hulls: list[ConvexHull] = []
        for i in range(len(skeleton) - 1):
            n0, n1 = skeleton[i], skeleton[i + 1]
            hull = ConvexHull(n0.pos, n1.pos)

            for cap in capsules:
                # Skip capsules belonging to the gate this segment is threading
                # through; otherwise the corridor would shrink to nothing. The
                # skeleton is already routed cleanly through the gate opening.
                if cap.is_gate and (
                    np.linalg.norm(cap.a - n0.pos) < 1.0 or np.linalg.norm(cap.a - n1.pos) < 1.0
                ):
                    continue

                c1, c2 = segment_segment_closest(n0.pos, n1.pos, cap.a, cap.b)
                vec = c1 - c2
                d = np.linalg.norm(vec)

                if d > 1e-5:
                    n = vec / d
                else:
                    seg = n1.pos - n0.pos
                    perp = (
                        np.array([-seg[1], seg[0], 0.0])
                        if np.linalg.norm(seg[:2]) > 1e-5
                        else np.array([1.0, 0.0, 0.0])
                    )
                    n = perp / np.linalg.norm(perp)

                eff_r = min(cap.radius, d - 0.005)
                plane_pt = c2 + n * eff_r
                hull.push_plane(-n, plane_pt)

            hulls.append(hull)
        return hulls

    def _solve_qp(
        self, skeleton: list[WayNode], corridors: list[ConvexHull], vel_now: NDArray
    ) -> NDArray:
        """Solve the convex QP for B-spline control points inside the corridors."""
        n_seg = len(corridors)
        per_seg = self.ctrl_pts_per_seg

        # Adapt control-point density on the (possibly very short) first segment
        if len(skeleton) > 1:
            d0 = np.linalg.norm(skeleton[1].pos - skeleton[0].pos)
            if d0 < 0.25:
                first_seg_pts = 1
            elif d0 < 0.50:
                first_seg_pts = 2
            elif d0 < 0.75:
                first_seg_pts = 3
            else:
                first_seg_pts = 4
        else:
            first_seg_pts = 1

        rest_seg_pts = per_seg
        n_ctrl = first_seg_pts + (n_seg - 1) * rest_seg_pts

        P = cp.Variable((n_ctrl, 3))
        cons: list = []

        ref_list = []
        for i in range(n_seg):
            k = first_seg_pts if i == 0 else rest_seg_pts
            for j in range(k):
                pt = corridors[i].a + (j / k) * (corridors[i].b_pt - corridors[i].a)
                ref_list.append(pt)
        ref_pts = np.array(ref_list)

        idx = 0
        for s, hull in enumerate(corridors):
            A = np.array(hull.A)
            b = np.array(hull.b)
            k = first_seg_pts if s == 0 else rest_seg_pts
            for _ in range(k):
                cons.append(A @ P[idx] <= b)
                idx += 1

        cons.append(P[-1] == skeleton[-1].pos)

        # Map skeleton-index -> control-point-index
        skel_to_cp = [0]
        idx = first_seg_pts
        for s in range(1, n_seg):
            skel_to_cp.append(idx)
            idx += rest_seg_pts
        skel_to_cp.append(n_ctrl - 1)

        for i in range(1, len(skeleton) - 1):
            if skeleton[i].is_gate:
                gi = skel_to_cp[i]
                cons.append(P[gi] == skeleton[i].pos)

        cost = (
            self.K_SMOOTH_VEL * cp.sum_squares(cp.diff(P, axis=0))
            + self.K_SMOOTH_ACC * cp.sum_squares(cp.diff(P, k=2, axis=0))
            + self.K_SMOOTH_JERK * cp.sum_squares(cp.diff(P, k=3, axis=0))
            + self.K_CENTERLINE * cp.sum_squares(P - ref_pts)
        )

        # Anchor the spline start to the drone's current state
        cost += self.K_START_POS * cp.sum_squares(P[0] - self._anchor_pos)

        if np.linalg.norm(vel_now) > 0.1:
            p_pred = self._anchor_pos + vel_now * self.LOOKAHEAD_DT
            cost += self.K_START_VEL * cp.sum_squares(P[1] - p_pred)
        else:
            cost += self.K_START_POS * cp.sum_squares(P[1] - self._anchor_pos)

        for i in range(1, len(skeleton) - 1):
            if skeleton[i].is_gate:
                gi = skel_to_cp[i]
                normal = skeleton[i].normal

                pre_authored = i - 1 >= 0 and skeleton[i - 1].is_authored
                post_authored = i + 1 < len(skeleton) and skeleton[i + 1].is_authored

                # Symmetric pass: gate point sits midway between its neighbors.
                # Skipped when either side is authored off-axis (override pre-anchor
                # or exit-detour) — symmetry would mirror the off-axis design onto
                # the other side and fight the intended geometry.
                if not (pre_authored or post_authored) and 0 <= gi - 1 and gi + 1 < n_ctrl:
                    cons.append(P[gi - 1] + P[gi + 1] == 2 * P[gi])

                # Pull approach control point onto the gate normal axis,
                # unless the adjacent skeleton waypoint was placed authored
                # (off-axis pre-override) — in which case we want the QP to
                # respect that placement, not pull it back onto the axis.
                if gi - 1 >= 0 and not pre_authored:
                    delta = P[gi - 1] - skeleton[i].pos
                    along = cp.reshape(delta @ normal, (1,), order="C") * normal
                    cost += self.K_GATE_AXIS * cp.sum_squares(delta - along)
                if gi + 1 < n_ctrl and not post_authored:
                    delta = P[gi + 1] - skeleton[i].pos
                    along = cp.reshape(delta @ normal, (1,), order="C") * normal
                    cost += self.K_GATE_AXIS * cp.sum_squares(delta - along)

        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            pass

        if P.value is None:
            print("Warning: corridor QP infeasible. Falling back to centerline reference.")
            return ref_pts[:n_ctrl]

        return P.value

    def _build_skeleton(self, pos_now: NDArray) -> list[WayNode]:
        """Apply the hardcoded gate-transition template to current gate poses."""
        gate_normals = R.from_quat(self.gates_quat).apply([1.0, 0.0, 0.0])
        path: list[WayNode] = [WayNode(pos_now, False, None, None, None)]
        empty_tr = GateTransition(0.0, (), None)

        def lookup(i: int) -> GateTransition:
            return LEVEL2_TRANSITIONS[i] if 0 <= i < len(LEVEL2_TRANSITIONS) else empty_tr

        for i in range(self.target_gate_idx, len(self.gates_pos)):
            pos = self.gates_pos[i]
            normal = gate_normals[i].copy()
            rot = R.from_quat(self.gates_quat[i])
            right = rot.apply([0.0, 1.0, 0.0])
            up = rot.apply([0.0, 0.0, 1.0])

            tr = lookup(i)

            # Pre-anchor: predecessor's next_pre_override takes priority
            prev_override = lookup(i - 1).next_pre_override if i >= 1 else None
            if prev_override is not None:
                fwd, lat, dz = prev_override
                pre = pos + normal * fwd + right * lat + np.array([0.0, 0.0, dz])
                pre_authored = True
            else:
                pre = pos - normal * self.anchor_gap
                pre_authored = False

            if abs(tr.entry_swing_lat) > 1e-6:
                path.append(WayNode(pos + right * tr.entry_swing_lat, False, None, None, None))

            if np.dot(pos - path[-1].pos, normal) > 0.05:
                path.append(WayNode(pre, False, None, None, None, pre_authored))

            # Gate centers always count as authored — they are fixed targets, the
            # deflection should never insert waypoints adjacent to them.
            path.append(WayNode(pos, True, normal, right, up, True))

            if tr.exit_detour:
                for fwd, lat, dz in tr.exit_detour:
                    wp = pos + normal * fwd + right * lat + np.array([0.0, 0.0, dz])
                    path.append(WayNode(wp, False, None, None, None, True))
            else:
                path.append(WayNode(pos + normal * self.anchor_gap, False, None, None, None))

        pole_circles: list[tuple[NDArray, float]] = []
        for p in self.obstacles_pos:
            pole_circles.append((p[:2], self.pole_radius + 0.15))
        gate_bar_circles: list[tuple[NDArray, float]] = []
        for p, q in zip(self.gates_pos, self.gates_quat):
            rot_g = R.from_quat(q)
            right_g = rot_g.apply([0.0, 1.0, 0.0])
            bar_dist = 0.28
            r_keep = 0.08 + 0.10
            gate_bar_circles.append(((p - right_g * bar_dist)[:2], r_keep))
            gate_bar_circles.append(((p + right_g * bar_dist)[:2], r_keep))

        smoothed = path
        for _ in range(3):
            nxt_path = [smoothed[0]]
            for i in range(1, len(smoothed)):
                prev_node = nxt_path[-1]
                curr_node = smoothed[i]
                prev_pt = prev_node.pos
                curr_pt = curr_node.pos
                ab = curr_pt[:2] - prev_pt[:2]
                len_sq = float(np.dot(ab, ab))

                if prev_node.is_authored and curr_node.is_authored:
                    circles = pole_circles
                else:
                    circles = pole_circles + gate_bar_circles

                if len_sq > 1e-6:
                    earliest = 1.0
                    detour: WayNode | None = None
                    for c, r_safe in circles:
                        t = max(0.0, min(1.0, np.dot(c - prev_pt[:2], ab) / len_sq))
                        proj = prev_pt[:2] + t * ab
                        d = np.linalg.norm(proj - c)
                        if d < r_safe and t < earliest:
                            earliest = t
                            push_dir = (
                                (proj - c) / d
                                if d > 1e-6
                                else np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
                            )
                            detour_xy = c + push_dir * (r_safe + 0.20)
                            detour_z = prev_pt[2] + t * (curr_pt[2] - prev_pt[2])
                            cand = np.array([detour_xy[0], detour_xy[1], detour_z])
                            if (
                                np.linalg.norm(cand - prev_pt) > 0.3
                                and np.linalg.norm(cand - curr_pt) > 0.3
                            ):
                                detour = WayNode(cand, False, None, None, None)
                    if detour is not None:
                        nxt_path.append(detour)
                nxt_path.append(smoothed[i])
            smoothed = nxt_path

        return smoothed

    # -------------------------------------------------------------- bookkeeping
    def _check_environment_updates(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Detect gate crossings and pose changes; replan if needed."""
        pos = obs["pos"]
        vel = obs.get("vel", np.zeros(3))
        if self._prev_pos is None:
            self._prev_pos = pos.copy()

        self._check_gate_crossed(pos)
        if self._check_objects_moved(obs) and self.target_gate_idx < len(self.gates_pos):
            self._build_trajectory(pos, vel)

        self._prev_pos = pos.copy()

    def _check_gate_crossed(self, pos_now: NDArray) -> bool:
        if self.target_gate_idx >= len(self.gates_pos):
            return False

        gpos = self.gates_pos[self.target_gate_idx]
        normal = R.from_quat(self.gates_quat[self.target_gate_idx]).apply([1.0, 0.0, 0.0])

        d_prev = np.dot(self._prev_pos - gpos, normal)
        d_curr = np.dot(pos_now - gpos, normal)

        if d_prev <= 0.0 < d_curr:
            cross = self._prev_pos + (d_prev / (d_prev - d_curr)) * (pos_now - self._prev_pos)
            if np.linalg.norm(cross - gpos) < (self.gate_outer_w / 2.0 + 0.40):
                self.target_gate_idx += 1
                return True
        return False

    def _check_objects_moved(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        moved = False
        new_gates = obs["gates_pos"]
        if (
            len(self.gates_pos) > 0
            and np.max(np.linalg.norm(new_gates - self.gates_pos, axis=1)) > 0.05
        ):
            self.gates_pos = new_gates.copy()
            self.gates_quat = obs["gates_quat"].copy()
            moved = True

        new_obs = obs.get("obstacles_pos", np.array([]))
        if len(new_obs) != len(self.obstacles_pos) or (
            len(new_obs) > 0 and np.max(np.linalg.norm(new_obs - self.obstacles_pos, axis=1)) > 0.05
        ):
            self.obstacles_pos = new_obs.copy()
            moved = True

        return moved

    # ----------------------------------------------------------------- runtime
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Sample the spline reference at the current tick.

        Args:
            obs: Observation dictionary.
            info: Optional info dictionary.

        Returns:
            Concatenation of desired pos, vel, acc, yaw, and zeros.
        """
        self._check_environment_updates(obs)

        t = min(self._spline_tick / self._freq, self._t_total)
        if t >= self._t_total and self.target_gate_idx >= len(self.gates_pos):
            self._finished = True

        u = t / self._t_total if self._t_total > 0 else 0.0
        du = 1.0 / self._t_total if self._t_total > 0 else 0.0

        pos = self._des_pos_spline(u)
        vel = self._des_pos_spline.derivative(nu=1)(u) * du
        acc = self._des_pos_spline.derivative(nu=2)(u) * (du**2)

        yaw = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.1 else 0.0

        return np.concatenate((pos, vel, acc, [yaw], np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance internal counters; return True when the plan is finished.

        Args:
            action: Action just executed.
            obs: Observation dictionary.
            reward: Reward signal.
            terminated: Episode termination flag.
            truncated: Episode truncation flag.
            info: Environment info dictionary.

        Returns:
            True once the spline has been fully consumed and all gates passed.
        """
        self._tick += 1
        self._spline_tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset per-episode state."""
        self._tick = 0
        self._spline_tick = 0
        self._finished = False
        self.target_gate_idx = 0
        self._prev_pos = None

    def render_callback(self, sim: Sim) -> None:
        """Draw the planned spline trajectory.

        Args:
            sim: Active simulator instance.
        """
        if not hasattr(self, "_des_pos_spline"):
            return
        draw_line(sim, self._des_pos_spline(np.linspace(0.0, 1.0, 100)), rgba=(0.0, 1.0, 0.0, 1.0))
