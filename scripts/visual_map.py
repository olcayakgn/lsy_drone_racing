"""Visualize (and optionally edit) a race-track config as a 2D top-down map.

This script plots:
- Drone starting position (from ``[[env.track.drones]]``)
- Gates (from ``[[env.track.gates]]``) with approximate orientation arrows (yaw)
- Poles/obstacles (from ``[[env.track.obstacles]]``)
- Optional safety-limits rectangle (from ``[env.track.safety_limits]``)

Run:
    python scripts/visual_map.py --config real_track_experiment.toml

Edit + save:
    python scripts/visual_map.py --config real_track_experiment.toml --edit --out my_track.toml

Edit + save (alias):
    python scripts/visual_map.py --config real_track_experiment.toml --edit --name my_track.toml

Editor controls (when ``--edit``):
    - Left click: select nearest object (gate / pole / start)
    - Drag: move selected object in the X-Y plane
    - E / Shift+E: rotate selected gate (yaw) by +/- 5 deg
    - Mouse wheel: rotate selected gate (yaw) by +/- 2 deg per scroll
    - S: save to ``--out`` (or a default name if omitted)
    - Esc: clear selection

Notes:
- The plot is intended for *planning* in the X-Y plane; Z is only shown in labels.
- In the provided configs, "poles" appear under ``env.track.obstacles``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

if TYPE_CHECKING:
    from matplotlib.backend_bases import Event

_DEFAULT_MOVE_PICK_RADIUS_M = 0.35
_GATE_ROT_STEP_RAD = math.radians(5.0)
_GATE_ROT_WHEEL_STEP_RAD = math.radians(2.0)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file into a plain dict."""
    try:
        import tomllib  # py>=3.11

        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        import toml  # type: ignore

        return toml.loads(path.read_text(encoding="utf-8"))


def _resolve_config_path(config_arg: str) -> Path:
    """Resolve --config value to an existing .toml file.

    Accepts:
    - an absolute/relative path
    - a bare filename that is searched under repo_root/config/
    - a filename without .toml extension
    """
    p = Path(config_arg)
    repo_root = Path(__file__).resolve().parents[1]

    candidates: list[Path] = []

    if p.suffix != ".toml":
        candidates.append(p.with_suffix(".toml"))
    candidates.append(p)

    # Also try under repo_root/config
    for c in list(candidates):
        if c.is_absolute():
            continue
        candidates.append(repo_root / "config" / c)

    for c in candidates:
        if c.exists() and c.is_file() and c.suffix == ".toml":
            return c

    tried = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Could not find config TOML. Tried:\n{tried}")


def _get_nested(dct: dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _xy(pos: Any) -> tuple[float, float]:
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        raise ValueError(f"Expected pos=[x,y,...], got: {pos!r}")
    return float(pos[0]), float(pos[1])


def _yaw_from_rpy(rpy: Any) -> float:
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
        return float(rpy[2])
    return 0.0


def _toml_dumps(data: dict[str, Any]) -> str:
    """Serialize to TOML.

    Uses the third-party `toml` package for dumping because stdlib `tomllib`
    does not provide dump/encode functions.
    """
    import toml  # type: ignore

    return toml.dumps(data)


def _default_out_path(config_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "config"
    return out_dir / f"{config_path.stem}_edited.toml"


def _distance2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def plot_map(
    config_path: Path,
    *,
    show: bool = True,
    edit: bool = False,
    out_path: Path | None = None,
    edit_start: bool = True,
) -> None:
    """Plot (and optionally interactively edit) a track config as a 2D top-down map."""
    cfg = _load_toml(config_path)

    track = _get_nested(cfg, "env", "track")
    if not isinstance(track, dict):
        raise KeyError("Missing [env.track] section in config")

    drones = track.get("drones", [])
    gates = track.get("gates", [])

    # In this repo's real-track configs, "poles" are encoded as obstacles.
    poles = track.get("obstacles", track.get("poles", []))

    if not isinstance(drones, list) or not isinstance(gates, list) or not isinstance(poles, list):
        raise TypeError("Expected env.track.{drones,gates,obstacles} to be arrays of tables")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Track map (top-down): {config_path.name}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)

    # ---- safety limits (optional) ----
    lim = track.get("safety_limits", {})
    safety_xlim: tuple[float, float] | None = None
    safety_ylim: tuple[float, float] | None = None
    if isinstance(lim, dict) and "pos_limit_low" in lim and "pos_limit_high" in lim:
        low = lim.get("pos_limit_low")
        high = lim.get("pos_limit_high")
        if (
            isinstance(low, (list, tuple))
            and isinstance(high, (list, tuple))
            and len(low) >= 2
            and len(high) >= 2
        ):
            x0, y0 = float(low[0]), float(low[1])
            x1, y1 = float(high[0]), float(high[1])
            # Normalize order just in case a config swaps low/high.
            xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
            ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
            safety_xlim = (xmin, xmax)
            safety_ylim = (ymin, ymax)
            w, h = xmax - xmin, ymax - ymin
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    fill=False,
                    edgecolor="gray",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
            )
            ax.text(xmin, ymin, "safety limits", color="gray", fontsize=9, va="bottom")

    # ---- poles/obstacles ----
    pole_artists: list[tuple[Circle, Circle]] = []
    pole_labels = []
    for i, pole in enumerate(poles):
        if not isinstance(pole, dict) or "pos" not in pole:
            pole_artists.append(
                (Circle((0.0, 0.0), 0.0, visible=False), Circle((0.0, 0.0), 0.0, visible=False))
            )
            pole_labels.append(None)
            continue
        x, y = _xy(pole["pos"])
        c1 = Circle((x, y), 0.05, color="k", alpha=0.85)
        c2 = Circle((x, y), 0.15, color="k", fill=False, alpha=0.25, linewidth=1.0)
        ax.add_patch(c1)
        ax.add_patch(c2)
        lab = ax.text(x, y + 0.18, f"P{i}", color="k", ha="center", fontsize=9)
        pole_artists.append((c1, c2))
        pole_labels.append(lab)

    # ---- gates ----
    gate_artists: list[dict[str, Any]] = []
    for gi, gate in enumerate(gates):
        if not isinstance(gate, dict) or "pos" not in gate:
            gate_artists.append({"lines": [], "arrow": None, "label": None})
            continue

        pos = gate["pos"]
        x, y = _xy(pos)
        z = float(pos[2]) if isinstance(pos, (list, tuple)) and len(pos) >= 3 else 0.0
        yaw = _yaw_from_rpy(gate.get("rpy"))

        col = f"C{gi % 10}"

        # Create placeholder artists we can update in edit mode.
        lines = []
        for half, lw, alpha in ((0.36, 1.5, 0.4), (0.20, 5.0, 0.9)):
            (ln,) = ax.plot([x, x], [y, y], color=col, lw=lw, alpha=alpha)
            lines.append((ln, half))

        arrow = ax.arrow(
            x,
            y,
            0.0,
            0.0,
            head_width=0.08,
            head_length=0.08,
            color=col,
            alpha=0.9,
            length_includes_head=True,
        )
        label = ax.text(
            x,
            y + 0.14,
            f"G{gi} (z={z:.2f})",
            color=col,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

        gate_artists.append({"lines": lines, "arrow": arrow, "label": label, "color": col})

        # Draw initial geometry.
        x_axis = (math.cos(yaw), math.sin(yaw))
        y_axis = (-math.sin(yaw), math.cos(yaw))
        for ln, half in lines:
            p1 = (x - half * y_axis[0], y - half * y_axis[1])
            p2 = (x + half * y_axis[0], y + half * y_axis[1])
            ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        arrow.set_visible(False)
        arrow = ax.arrow(
            x,
            y,
            0.35 * x_axis[0],
            0.35 * x_axis[1],
            head_width=0.08,
            head_length=0.08,
            color=col,
            alpha=0.9,
            length_includes_head=True,
        )
        gate_artists[-1]["arrow"] = arrow

    # ---- start position ----
    start_artist: dict[str, Any] | None = None
    if drones:
        start = drones[0]
        if isinstance(start, dict) and "pos" in start:
            sx, sy = _xy(start["pos"])
            syaw = _yaw_from_rpy(start.get("rpy"))
            (start_pt,) = ax.plot(sx, sy, "ks", ms=10)
            start_arrow = ax.arrow(
                sx,
                sy,
                0.25 * math.cos(syaw),
                0.25 * math.sin(syaw),
                head_width=0.07,
                head_length=0.07,
                color="k",
                alpha=0.9,
                length_includes_head=True,
            )
            start_label = ax.text(sx, sy - 0.18, "Start", color="k", ha="center", fontsize=10)
            start_artist = {"pt": start_pt, "arrow": start_arrow, "label": start_label}

    # ---- auto limits ----
    xs: list[float] = []
    ys: list[float] = []

    def _collect(item_list: list[Any]) -> None:
        for it in item_list:
            if isinstance(it, dict) and "pos" in it:
                try:
                    x, y = _xy(it["pos"])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)

    _collect(poles)
    _collect(gates)
    _collect(drones)

    # If safety limits exist, always use them for plot bounds (X-Y).
    if safety_xlim is not None and safety_ylim is not None:
        ax.set_xlim(*safety_xlim)
        ax.set_ylim(*safety_ylim)
    elif xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span = max(maxx - minx, maxy - miny)
        margin = max(0.5, 0.15 * span)
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="k", linestyle="None", markersize=8, label="Start"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                linestyle="None",
                markersize=6,
                label="Pole/obstacle",
            ),
        ],
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )

    if not edit:
        fig.tight_layout()
        if show:
            plt.show()
        return

    # Leave room for the big save button.
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])

    # -----------------
    # Interactive editor
    # -----------------
    help_text = (
        "Edit mode: click-select, drag-move | E/Shift+E rotate | wheel rotate | S save | Esc clear"
    )
    fig.text(0.01, 0.02, help_text, fontsize=9, alpha=0.85)

    if out_path is None:
        out_path = _default_out_path(config_path)

    selected: dict[str, Any] = {"kind": None, "idx": None}
    dragging = {"active": False}

    def _gate_pos_yaw(gi: int) -> tuple[float, float, float]:
        gate = gates[gi]
        assert isinstance(gate, dict)
        pos = gate.get("pos")
        x, y = _xy(pos)
        yaw = _yaw_from_rpy(gate.get("rpy"))
        return x, y, yaw

    def _set_gate_xy(gi: int, x: float, y: float) -> None:
        gate = gates[gi]
        assert isinstance(gate, dict)
        pos = gate.get("pos")
        if isinstance(pos, list) and len(pos) >= 2:
            pos[0] = float(x)
            pos[1] = float(y)
        elif isinstance(pos, tuple) and len(pos) >= 2:
            gate["pos"] = [float(x), float(y), *list(pos[2:])]
        else:
            gate["pos"] = [float(x), float(y), 0.0]

    def _set_gate_yaw(gi: int, yaw: float) -> None:
        gate = gates[gi]
        assert isinstance(gate, dict)
        rpy = gate.get("rpy")
        if isinstance(rpy, list) and len(rpy) >= 3:
            rpy[2] = float(yaw)
        elif isinstance(rpy, tuple) and len(rpy) >= 3:
            gate["rpy"] = [float(rpy[0]), float(rpy[1]), float(yaw)]
        else:
            gate["rpy"] = [0.0, 0.0, float(yaw)]

    def _update_gate_artists(gi: int) -> None:
        if gi >= len(gate_artists):
            return
        if not (isinstance(gates[gi], dict) and "pos" in gates[gi]):
            return

        x, y, yaw = _gate_pos_yaw(gi)
        x_axis = (math.cos(yaw), math.sin(yaw))
        y_axis = (-math.sin(yaw), math.cos(yaw))
        artists = gate_artists[gi]

        for ln, half in artists["lines"]:
            p1 = (x - half * y_axis[0], y - half * y_axis[1])
            p2 = (x + half * y_axis[0], y + half * y_axis[1])
            ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])

        # Replace arrow (matplotlib arrows are not easily mutable)
        old_arrow = artists.get("arrow")
        if old_arrow is not None:
            try:
                old_arrow.remove()
            except Exception:
                pass
        col = artists.get("color", f"C{gi % 10}")
        artists["arrow"] = ax.arrow(
            x,
            y,
            0.35 * x_axis[0],
            0.35 * x_axis[1],
            head_width=0.08,
            head_length=0.08,
            color=col,
            alpha=0.9,
            length_includes_head=True,
        )

        lab = artists.get("label")
        if lab is not None:
            pos = gates[gi]["pos"]
            z = float(pos[2]) if isinstance(pos, (list, tuple)) and len(pos) >= 3 else 0.0
            lab.set_position((x, y + 0.14))
            lab.set_text(f"G{gi} (z={z:.2f})")

    def _set_pole_xy(pi: int, x: float, y: float) -> None:
        pole = poles[pi]
        assert isinstance(pole, dict)
        pos = pole.get("pos")
        if isinstance(pos, list) and len(pos) >= 2:
            pos[0] = float(x)
            pos[1] = float(y)
        elif isinstance(pos, tuple) and len(pos) >= 2:
            pole["pos"] = [float(x), float(y), *list(pos[2:])]
        else:
            pole["pos"] = [float(x), float(y), 0.0]

    def _update_pole_artists(pi: int) -> None:
        if pi >= len(pole_artists):
            return
        pole = poles[pi]
        if not (isinstance(pole, dict) and "pos" in pole):
            return
        x, y = _xy(pole["pos"])
        c1, c2 = pole_artists[pi]
        c1.center = (x, y)
        c2.center = (x, y)
        lab = pole_labels[pi]
        if lab is not None:
            lab.set_position((x, y + 0.18))

    def _set_start_xy(x: float, y: float) -> None:
        if not drones:
            return
        start = drones[0]
        if not isinstance(start, dict):
            return
        pos = start.get("pos")
        if isinstance(pos, list) and len(pos) >= 2:
            pos[0] = float(x)
            pos[1] = float(y)
        elif isinstance(pos, tuple) and len(pos) >= 2:
            start["pos"] = [float(x), float(y), *list(pos[2:])]
        else:
            start["pos"] = [float(x), float(y), 0.0]

    def _set_start_yaw(yaw: float) -> None:
        if not drones:
            return
        start = drones[0]
        if not isinstance(start, dict):
            return
        rpy = start.get("rpy")
        if isinstance(rpy, list) and len(rpy) >= 3:
            rpy[2] = float(yaw)
        elif isinstance(rpy, tuple) and len(rpy) >= 3:
            start["rpy"] = [float(rpy[0]), float(rpy[1]), float(yaw)]
        else:
            start["rpy"] = [0.0, 0.0, float(yaw)]

    def _update_start_artists() -> None:
        if start_artist is None or not drones:
            return
        start = drones[0]
        if not isinstance(start, dict) or "pos" not in start:
            return
        sx, sy = _xy(start["pos"])
        syaw = _yaw_from_rpy(start.get("rpy"))
        start_artist["pt"].set_data([sx], [sy])
        try:
            start_artist["arrow"].remove()
        except Exception:
            pass
        start_artist["arrow"] = ax.arrow(
            sx,
            sy,
            0.25 * math.cos(syaw),
            0.25 * math.sin(syaw),
            head_width=0.07,
            head_length=0.07,
            color="k",
            alpha=0.9,
            length_includes_head=True,
        )
        start_artist["label"].set_position((sx, sy - 0.18))

    def _select_nearest(x: float, y: float) -> None:
        best = {"kind": None, "idx": None, "d2": float("inf")}

        # gates
        for gi, gate in enumerate(gates):
            if not (isinstance(gate, dict) and "pos" in gate):
                continue
            gx, gy = _xy(gate["pos"])
            d2 = _distance2((x, y), (gx, gy))
            if d2 < best["d2"]:
                best = {"kind": "gate", "idx": gi, "d2": d2}

        # poles
        for pi, pole in enumerate(poles):
            if not (isinstance(pole, dict) and "pos" in pole):
                continue
            px, py = _xy(pole["pos"])
            d2 = _distance2((x, y), (px, py))
            if d2 < best["d2"]:
                best = {"kind": "pole", "idx": pi, "d2": d2}

        # start
        if edit_start and drones and isinstance(drones[0], dict) and "pos" in drones[0]:
            sx, sy = _xy(drones[0]["pos"])
            d2 = _distance2((x, y), (sx, sy))
            if d2 < best["d2"]:
                best = {"kind": "start", "idx": 0, "d2": d2}

        if best["kind"] is None:
            selected["kind"], selected["idx"] = None, None
            return

        if best["d2"] <= _DEFAULT_MOVE_PICK_RADIUS_M**2:
            selected["kind"], selected["idx"] = best["kind"], best["idx"]
        else:
            selected["kind"], selected["idx"] = None, None

    def _rotate_selected(delta: float) -> None:
        if selected["kind"] == "gate" and selected["idx"] is not None:
            gi = int(selected["idx"])
            _, _, yaw = _gate_pos_yaw(gi)
            _set_gate_yaw(gi, yaw + delta)
            _update_gate_artists(gi)
            fig.canvas.draw_idle()
        elif selected["kind"] == "start" and selected["idx"] is not None and edit_start:
            start = drones[0] if drones else None
            if isinstance(start, dict):
                yaw = _yaw_from_rpy(start.get("rpy"))
                _set_start_yaw(yaw + delta)
                _update_start_artists()
                fig.canvas.draw_idle()

    def _save() -> None:
        nonlocal out_path
        assert out_path is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_toml_dumps(cfg), encoding="utf-8")
        print(f"Saved edited track to: {out_path}")

    # Big save button
    btn_ax = fig.add_axes([0.35, 0.035, 0.30, 0.06])
    save_btn = Button(btn_ax, "SAVE TRACK")
    try:
        save_btn.label.set_fontsize(14)
        save_btn.label.set_fontweight("bold")
    except Exception:
        pass
    save_btn.on_clicked(lambda _event: _save())

    def on_press(event: Event) -> None:
        if event.inaxes != ax:
            return
        if event.key == "escape":
            selected["kind"], selected["idx"] = None, None
            dragging["active"] = False
            fig.canvas.draw_idle()
            return
        # Note: we avoid binding 'q' because some Matplotlib backends map it to quit.
        if event.key == "e":
            _rotate_selected(_GATE_ROT_STEP_RAD)
            return
        if event.key == "E":
            _rotate_selected(-_GATE_ROT_STEP_RAD)
            return
        if event.key in ("s", "S"):
            _save()
            return

    def on_scroll(event: Event) -> None:
        if event.inaxes != ax:
            return
        if getattr(event, "step", 0) == 0:
            return
        _rotate_selected(_GATE_ROT_WHEEL_STEP_RAD * float(event.step))

    def on_button_press(event: Event) -> None:
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        _select_nearest(float(event.xdata), float(event.ydata))
        dragging["active"] = selected["kind"] is not None
        fig.canvas.draw_idle()

    def on_button_release(event: Event) -> None:
        if event.button != 1:
            return
        dragging["active"] = False

    def on_motion(event: Event) -> None:
        if not dragging["active"]:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if selected["kind"] == "gate" and selected["idx"] is not None:
            gi = int(selected["idx"])
            _set_gate_xy(gi, x, y)
            _update_gate_artists(gi)
            fig.canvas.draw_idle()
        elif selected["kind"] == "pole" and selected["idx"] is not None:
            pi = int(selected["idx"])
            _set_pole_xy(pi, x, y)
            _update_pole_artists(pi)
            fig.canvas.draw_idle()
        elif selected["kind"] == "start" and edit_start:
            _set_start_xy(x, y)
            _update_start_artists()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("button_release_event", on_button_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    if show:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and plot the track map; return a process exit code."""
    parser = argparse.ArgumentParser(description="Plot a 2D top-down track map from a TOML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Config TOML path or filename under ./config (e.g. real_track_experiment.toml)",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Enable interactive editing (drag/move objects, rotate gates, save)",
    )
    parser.add_argument(
        "--out",
        "--name",
        dest="out",
        default=None,
        help="Output TOML filename (saved under ./config/). Use --name as an alias.",
    )
    parser.add_argument(
        "--no-edit-start", action="store_true", help="Do not allow editing the start pose."
    )

    args = parser.parse_args(argv)

    try:
        config_path = _resolve_config_path(args.config)

        out_path: Path | None = None
        if args.out:
            out_candidate = Path(args.out)
            if out_candidate.suffix != ".toml":
                out_candidate = out_candidate.with_suffix(".toml")
            # Always save into the repo's config/ folder (ignore provided directories).
            out_path = Path(__file__).resolve().parents[1] / "config" / out_candidate.name

        plot_map(
            config_path,
            show=True,
            edit=bool(args.edit),
            out_path=out_path,
            edit_start=not bool(args.no_edit_start),
        )
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
