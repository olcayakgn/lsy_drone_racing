"""Debug plotting harness for drone-racing sim runs.

Runs the simulation like ``scripts/sim.py`` but records the drone trajectory
and the TRUE track layout (read from the environment internals, not the
sensor-masked observation), then writes one annotated PNG per run for visual
debugging of crashes and speed analysis.

Each PNG has three panels:
    1. Top-down (x-y) view of the full path, colour-coded by which gate the
       drone was chasing, with true gates / obstacles overlaid and the
       controller's final planned route (if exposed) as a dashed line.
    2. Altitude (z) vs time, with gate heights marked and gate-pass events.
    3. Speed |v| vs time.

Run:
    pixi run python scripts/debug_plot.py --config level3.toml --n_runs 5

Output: one PNG per run in ``--out_dir`` (default ``/tmp/sim_plots``).
"""

from __future__ import annotations

import os

# scipy's array API must be enabled before scipy is first imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402

logger = logging.getLogger(__name__)


def _true_track(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the true gate/obstacle layout from the env internals.

    The observation masks gate/obstacle positions until the drone has sensed
    them; the env's own ``data`` always holds the real values.
    """
    u = env.unwrapped
    gates_pos = np.asarray(u.data.gates_pos[0], dtype=float)
    gates_quat = np.asarray(u.data.gates_quat[0], dtype=float)
    obstacles_pos = np.asarray(u.data.obstacles_pos[0], dtype=float)
    return gates_pos, gates_quat, obstacles_pos


def _extract_planned_route(ctrl) -> np.ndarray | None:
    """Return the controller's currently-planned route polyline, if exposed.

    Best-effort: try a few attribute names used by the in-repo controllers.
    Returns an (M, 3) array or None.
    """
    for attr in ("_active_route_points", "_last_ref_pos", "_cached_ref_pos"):
        route = getattr(ctrl, attr, None)
        if route is None:
            continue
        route = np.asarray(route, dtype=float)
        if route.ndim == 2 and route.shape[1] >= 3 and route.shape[0] >= 2:
            return route[:, :3]
    return None


def _plot_run(
    path: Path,
    run_idx: int,
    traj: np.ndarray,
    vel: np.ndarray,
    tgt: np.ndarray,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
    obstacles_pos: np.ndarray,
    start: np.ndarray,
    gates_passed: int,
    finished: bool,
    flight_time: float,
    outcome: str,
    planned_route: np.ndarray | None,
    env_freq: float,
) -> None:
    """Write the three-panel trajectory plot for one run."""
    n_gates = len(gates_pos)
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axz = fig.add_subplot(gs[0, 1])
    axv = fig.add_subplot(gs[0, 2])

    # ----- top-down x-y -----
    ax.set_title("top-down (x-y)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)

    for op in obstacles_pos:
        ax.add_patch(plt.Circle((op[0], op[1]), 0.05, color="k", alpha=0.8))
        ax.add_patch(plt.Circle((op[0], op[1]), 0.15, color="k", fill=False, alpha=0.3))

    for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        rot = Rot.from_quat(gq).as_matrix()
        x_axis, y_axis = rot[:, 0], rot[:, 1]  # normal, lateral opening dir
        col = f"C{gi}"
        # gate frame extent (0.72 m) thin, opening (0.40 m) thick
        for half, lw, alpha in ((0.36, 1.5, 0.4), (0.20, 5.0, 0.9)):
            p1 = gp[:2] - half * y_axis[:2]
            p2 = gp[:2] + half * y_axis[:2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=lw, alpha=alpha)
        # required pass direction (gate +x)
        ax.arrow(
            gp[0], gp[1], 0.35 * x_axis[0], 0.35 * x_axis[1],
            head_width=0.08, head_length=0.08, color=col, alpha=0.9,
        )
        ax.text(
            gp[0], gp[1] + 0.14, f"G{gi} (z={gp[2]:.2f})", color=col,
            ha="center", fontsize=10, fontweight="bold",
        )

    for g in range(-1, n_gates):
        mask = tgt == g
        if mask.any():
            col = "gray" if g < 0 else f"C{g}"
            label = "post-finish" if g < 0 else f"toward G{g}"
            ax.plot(traj[mask, 0], traj[mask, 1], ".", ms=3.5, color=col, label=label)

    if planned_route is not None and len(planned_route) >= 2:
        ax.plot(
            planned_route[:, 0], planned_route[:, 1],
            "--", color="magenta", lw=1.2, alpha=0.7, label="planned route (final)",
        )

    ax.plot(start[0], start[1], "ks", ms=10, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "rX", ms=16, label=f"end ({outcome})")
    ax.legend(fontsize=8, loc="upper right")

    # ----- altitude vs time -----
    t = np.arange(len(traj)) / env_freq
    axz.set_title("altitude (z) vs time")
    axz.set_xlabel("time [s]")
    axz.set_ylabel("z [m]")
    axz.grid(alpha=0.3)
    axz.plot(t, traj[:, 2], "-", color="navy", lw=1.5)
    for gi, gp in enumerate(gates_pos):
        axz.axhline(gp[2], color=f"C{gi}", ls="--", alpha=0.5)
        axz.text(t[-1], gp[2], f" G{gi}", color=f"C{gi}", va="center", fontsize=9)
    for c in np.where(np.diff(tgt) != 0)[0]:
        axz.axvline(t[c], color="green", ls=":", alpha=0.7)
    axz.text(
        0.02, 0.97, "green dotted = gate passed",
        transform=axz.transAxes, fontsize=8, va="top", color="green",
    )

    # ----- speed |v| vs time -----
    speed = np.linalg.norm(vel, axis=1)
    axv.set_title("speed |v| vs time")
    axv.set_xlabel("time [s]")
    axv.set_ylabel("|v| [m/s]")
    axv.grid(alpha=0.3)
    axv.plot(t, speed, "-", color="darkorange", lw=1.5)
    if speed.size:
        axv.axhline(float(np.mean(speed)), color="gray", ls=":", alpha=0.6,
                    label=f"mean {np.mean(speed):.2f}")
        axv.axhline(float(np.max(speed)), color="red", ls=":", alpha=0.4,
                    label=f"peak {np.max(speed):.2f}")
        axv.legend(fontsize=8, loc="lower right")
    for c in np.where(np.diff(tgt) != 0)[0]:
        axv.axvline(t[c], color="green", ls=":", alpha=0.7)

    fig.suptitle(
        f"run {run_idx}   gates passed: {gates_passed}/{n_gates}   "
        f"finished: {finished}   flight time: {flight_time:.2f} s   [{outcome}]",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def debug_plot(
    config: str = "level3.toml",
    controller: str | None = None,
    n_runs: int = 5,
    out_dir: str = "/tmp/sim_plots",
) -> None:
    """Run the sim and write one annotated trajectory PNG per episode."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_gates_total = len(cfg.env.track.gates)
    env_freq = float(cfg.env.freq)

    for run in range(1, n_runs + 1):
        obs, info = env.reset()
        gates_pos, gates_quat, obstacles_pos = _true_track(env)
        ctrl = controller_cls(obs, info, cfg)

        start = np.asarray(obs["pos"], dtype=float)
        traj = [start.copy()]
        vel = [np.asarray(obs["vel"], dtype=float)]
        tgt = [int(obs["target_gate"])]

        i = 0
        terminated = truncated = ctrl_done = False
        final_target = int(obs["target_gate"])
        while True:
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ctrl_done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            final_target = int(obs["target_gate"])
            if terminated:
                # On a crash the env warps the disabled drone away, so
                # obs["pos"] is no longer the crash location -- do not record
                # it. The trajectory ends at the last pre-crash position.
                break
            traj.append(np.asarray(obs["pos"], dtype=float))
            vel.append(np.asarray(obs["vel"], dtype=float))
            tgt.append(final_target)
            i += 1
            if truncated or ctrl_done:
                break

        planned_route = _extract_planned_route(ctrl)
        ctrl.episode_callback()
        flight_time = i / env_freq
        gp = final_target
        gates_passed = n_gates_total if gp == -1 else gp
        finished = gates_passed == n_gates_total
        outcome = "FINISHED" if finished else ("TIMEOUT" if truncated else "CRASH")
        ctrl.episode_reset()

        path = out / f"run_{run}.png"
        _plot_run(
            path, run,
            np.asarray(traj), np.asarray(vel), np.asarray(tgt),
            gates_pos, gates_quat, obstacles_pos,
            start, gates_passed, finished, flight_time, outcome,
            planned_route, env_freq,
        )
        print(
            f"run {run}: gates_passed={gates_passed} finished={finished} "
            f"time={flight_time:.2f}s {outcome} -> {path}"
        )

    env.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    fire.Fire(debug_plot)
