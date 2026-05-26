from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation as R

# Attempt to load yaml, handle failure gracefully if file is missing during local test
try:
    from lsy_drone_racing.control.spatial_utils.yaml_import import load_yaml
    CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")
except (ImportError, FileNotFoundError):
    CONSTANTS = {}


class GeometryEngine:
    """
    Handles geometric path planning, frame generation, and corridor constraint 
    creation for drone racing using Differential Geometry concepts.
    
    This engine generates a reference path using Cubic Splines, calculates a 
    Parallel Transport (Bishop) Frame to avoid singularities associated with 
    Frenet-Serret frames, and computes static linear constraints (corridors) 
    for MPC formulation.
    """

    def __init__(
        self,
        gates_pos: List[List[float]],
        gates_normal: List[List[float]],
        gates_y: List[List[float]],
        gates_z: List[List[float]],
        gate_size: float = 0.5,
        obstacles_pos: List[List[float]] = [],
        start_pos: List[float] = [-1.5, 0.75, 0.01],
        start_orient: List[float] = [0.0, 0.0, 0.0],
        obs: Optional[Dict[str, NDArray[np.floating]]] = None,
        info: Optional[Dict] = None,
        sim_config: Optional[Dict] = None,
    ):
        """
        Initialize the Geometry Engine.

        Args:
            gates_pos: Center positions of the gates (N x 3).
            gates_normal: Normal vectors of the gates (N x 3).
            gates_y: Lateral (Y) vectors of the gates (N x 3).
            gates_z: Vertical (Z) vectors of the gates (N x 3).
            gate_size: Dimension of the square gate.
            obstacles_pos: Center positions of cylindrical obstacles.
            start_pos: Initial position of the drone [x, y, z].
            start_orient: Initial Euler orientation [r, p, y].
            obs: Observation dictionary (optional context).
            info: Info dictionary (optional context).
            sim_config: Simulation configuration dictionary.
        """
        # --- Data Initialization ---
        self.gates_pos = np.array(gates_pos, dtype=np.float64)
        self.gates_normal = np.array(gates_normal, dtype=np.float64)
        self.gates_y = np.array(gates_y, dtype=np.float64)
        self.gates_z = np.array(gates_z, dtype=np.float64)
        self.obstacles_pos = np.array(obstacles_pos, dtype=np.float64)
        
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.start_orient = R.from_euler("xyz", start_orient).as_matrix()

        self.gate_size = gate_size
        self.obs = obs if obs is not None else {}
        self.info = info if info is not None else {}
        self.sim_config = sim_config if sim_config is not None else {}

        # --- Geometric Constants (Tunable) ---
        self.POLE_HEIGHT = 3.0
        self.SAFETY_RADIUS = 0.1
        self.MAX_LATERAL_WIDTH = 0.2
        self.CONTRACTION_LEN = 0.3          # Distance to contract constraints near obstacles
        self.CLEARANCE_RADIUS = 0.25        # Radius for virtual obstacle avoidance
        self.GATE_CONTRACTION_LEN = 0.3     # Distance to contract constraints near gates
        self.U_TURN_EXTENSION = 0.25        # Distance to fly straight after a reversal gate
        self.U_TURN_RADIUS = 0.35           # Radius for heuristic U-turn generation
        self.GATE_APPROACH_DIST = 0.5       # Distance for pre/post gate guidance

        # Debugging storage
        self.debug_dicts: List[Dict] = []

        # --- Pipeline Execution ---
        
        # 1. Generate Vectors
        self.gate_vectors = self._compute_gate_to_gate_vectors()

        # 2. Heuristic Waypoint Generation (Handle U-Turns)
        raw_waypoints = self._initialize_waypoints()
        
        # 3. Obstacle Avoidance (Modify Waypoints)
        self.waypoints = self._insert_obstacle_avoidance_waypoints(
            raw_waypoints, clearance_radius=self.CLEARANCE_RADIUS
        )

        # 4. Spline Generation
        self.spline = self._get_spline(self.waypoints)

        # 5. Bishop Frame Generation (Parallel Transport)
        # Higher density ensures smoother discrete constraints
        num_frame_points = int(max(10, self.total_length * 100))
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)

        # 6. Static Corridor Generation (Constraints)
        self.corridor_map = self._generate_static_corridor()

    # =========================================================================
    # Section 1: Waypoint & Path Generation
    # =========================================================================

    def _compute_gate_to_gate_vectors(self) -> List[NDArray[np.float64]]:
        """Computes normalized vectors between consecutive gates."""
        gate_vectors = []
        for i in range(1, len(self.gates_pos)):
            vec = self.gates_pos[i] - self.gates_pos[i - 1]
            norm = np.linalg.norm(vec)
            gate_vectors.append(vec / norm if norm > 1e-6 else vec)
        return gate_vectors

    def _initialize_waypoints(self) -> NDArray[np.float64]:
        """
        Initialize waypoints with heuristics for 180-degree reversals.
        
        Detects if the angle between the current gate normal and the vector 
        to the next gate is sharp (>120 deg). If so, it inserts "Balloon" 
        waypoints to force a wide U-turn rather than a snap turn.
        """
        waypoints = [self.start_pos]

        for idx, gate_pos in enumerate(self.gates_pos):
            # 1. Pre-Gate Alignment
            before_gate = gate_pos - self.gates_normal[idx] * self.GATE_APPROACH_DIST
            waypoints.append(before_gate)
            
            # 2. Gate Center
            waypoints.append(gate_pos)

            # 3. Reversal Detection
            is_reversal = False
            if idx < len(self.gates_pos) - 1:
                vec_to_next = self.gates_pos[idx + 1] - gate_pos
                angle = self.angle_between_vectors(vec_to_next, self.gates_normal[idx])
                
                # If angle > 120 degrees, the next gate is "behind" us
                if np.degrees(angle) > 120:
                    is_reversal = True

            if is_reversal:
                # print(f"[Geometry] Generating Reversal Balloon at Gate {idx}")
                
                # Extension: Fly straight out first
                extension_point = gate_pos + self.gates_normal[idx] * self.U_TURN_EXTENSION
                waypoints.append(extension_point)

                # Lateral Shift: Determine Left vs Right turn based on dot product
                vec_to_next = self.gates_pos[idx + 1] - gate_pos
                side_sign = np.sign(np.dot(vec_to_next, self.gates_y[idx]))
                if side_sign == 0: 
                    side_sign = 1.0 # Default to left
                
                # Balloon Point
                turn_point = extension_point + (self.gates_y[idx] * self.U_TURN_RADIUS * side_sign)
                waypoints.append(turn_point)

            else:
                # Standard Exit: Follow normal out
                after_gate = gate_pos + self.gates_normal[idx] * self.GATE_APPROACH_DIST
                waypoints.append(after_gate)

        return np.array(waypoints)

    def _insert_obstacle_avoidance_waypoints(
        self, waypoints: NDArray[np.float64], clearance_radius: float, num_eval_points: int = 1000
    ) -> NDArray[np.float64]:
        """
        Refines the path by pushing waypoints away from obstacles.
        
        It samples the current spline, checks for collision zones, and inserts
        intermediate waypoints shifted along the repulsion vector.
        """
        # Create temporary spline to sample path
        temp_spline = self._get_spline(waypoints)
        s_eval = np.linspace(0, self.total_length, num_eval_points)
        sampled_points = temp_spline(s_eval)
        
        # Iteratively refine points against all obstacles
        # Note: This updates sampled_points in place conceptually (though logic below simplifies this)
        current_points = sampled_points
        
        for obs in self.obstacles_pos:
            current_points = self._process_single_obstacle_avoidance(current_points, clearance_radius, obs)
            
        # We assume the output of this process acts as the new dense set of waypoints
        # Alternatively, one could downsample this list to reduce spline complexity.
        return current_points

    def _process_single_obstacle_avoidance(
        self, sampled_points: NDArray[np.float64], clearance_radius: float, obstacle_center: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Identifies segments where the path intersects an obstacle and bends the 
        path outward.
        """
        collision_free_points = []
        is_inside_zone = False
        entry_index = 0
        
        obstacle_xy = obstacle_center[:2]

        for i, point in enumerate(sampled_points):
            point_xy = point[:2]
            dist_xy = np.linalg.norm(obstacle_xy - point_xy)

            if dist_xy < clearance_radius:
                if not is_inside_zone:
                    # Entered collision zone
                    is_inside_zone = True
                    entry_index = i
            
            elif is_inside_zone:
                # Exited collision zone
                is_inside_zone = False
                exit_index = i

                # --- Calculation ---
                entry_pt = sampled_points[entry_index]
                exit_pt = sampled_points[exit_index]

                # Vectors relative to obstacle center
                entry_vec = entry_pt[:2] - obstacle_xy
                exit_vec = exit_pt[:2] - obstacle_xy

                # Bisector vector: Direction to push
                avoid_vec = entry_vec + exit_vec
                avoid_vec /= (np.linalg.norm(avoid_vec) + 1e-6)

                # Create new waypoint at edge of clearance
                new_pos_xy = obstacle_xy + avoid_vec * clearance_radius
                # Maintain altitude (average of entry/exit)
                new_pos_z = (entry_pt[2] + exit_pt[2]) / 2.0 
                
                new_avoid_wp = np.concatenate([new_pos_xy, [new_pos_z]])

                # We append the new waypoint instead of the colliding points
                # Note: This is a simplification; strictly we might want entry -> avoid -> exit
                collision_free_points.append(new_avoid_wp)
                
                # Append current safe point
                collision_free_points.append(point)

            else:
                # Safe point
                collision_free_points.append(point)

        return np.array(collision_free_points)

    def _get_spline(self, points: NDArray[np.float64]) -> CubicHermiteSpline:
        """Generates a Cubic Spline from a set of 3D points."""
        s_knots = self._create_s_knots(points)
        # Using CubicSpline as a wrapper (scipy handles Hermite implicitly or via BCs)
        # Ideally, if tangents are known, use CubicHermiteSpline. Here we let Scipy estimate tangents.
        spline = CubicSpline(s_knots, points)
        return spline

    def _create_s_knots(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes cumulative distance (arc length approximation) for spline knots."""
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.maximum(dists, 1e-6)  # Safety
        s_knots = np.concatenate(([0], np.cumsum(dists)))
        
        self.total_length = s_knots[-1]
        return s_knots

    # =========================================================================
    # Section 2: Reference Frame (Bishop Frame)
    # =========================================================================

    def _generate_parallel_transport_frame(self, num_points: int = 3000) -> Dict[str, Union[NDArray, List]]:
        """
        Generates the Bishop Frame (Parallel Transport Frame) along the spline.
        
        Unlike the Frenet-Serret frame, the Bishop frame is defined even when 
        curvature vanishes (straight lines) and does not twist excessively around 
        the tangent, making it ideal for drone corridors.
        """
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        # Initialize storage
        frames = {
            "s": s_eval,
            "pos": [], "t": [], "n1": [], "n2": [],
            "k1": [], "k2": [], "dk1": [], "dk2": [],
        }

        # --- Initial Frame ---
        # Tangent
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)

        # Normal vector initial guess (using Gravity [0,0,-1] reference)
        g_vec = np.array([0, 0, -1])
        
        # Handle singularity if start is perfectly vertical
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            # Project gravity onto plane orthogonal to tangent
            n2_0 = g_vec - np.dot(g_vec, t0) * t0

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0) # Right-hand rule

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0
        k1_list, k2_list = [], []

        # --- Propagate Frame ---
        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            
            # Curvature vector (2nd derivative)
            k_vec = self.spline(s, 2)
            
            # Decompose curvature into body frame components
            k1 = np.dot(k_vec, curr_n1)
            k2 = np.dot(k_vec, curr_n2)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            k1_list.append(k1)
            k2_list.append(k2)

            # Parallel Transport step
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                norm_next = np.linalg.norm(next_t)
                if norm_next > 1e-6:
                    next_t /= norm_next

                # Compute rotation needed to align curr_t with next_t
                axis = np.cross(curr_t, next_t)
                dot_prod = np.clip(np.dot(curr_t, next_t), -1.0, 1.0)
                angle = np.arccos(dot_prod)

                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1 = r_vec.apply(curr_n1)
                    next_n2 = r_vec.apply(curr_n2)
                else:
                    # Tangents are parallel, frame doesn't rotate
                    next_n1, next_n2 = curr_n1, curr_n2

                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        # Convert lists to arrays & compute derivatives of curvature
        frames["pos"] = np.array(frames["pos"])
        frames["t"] = np.array(frames["t"])
        frames["n1"] = np.array(frames["n1"])
        frames["n2"] = np.array(frames["n2"])
        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)

        return frames

    # =========================================================================
    # Section 3: Static Corridor Generation (Constraints)
    # =========================================================================

    def _generate_static_corridor(self) -> Dict[str, NDArray[np.float64]]:
        """
        Generates linear bound constraints (lb <= w1 <= ub) for the path.
        
        Projects obstacles and gates into the local curvilinear coordinate 
        system (s, n1, n2) and constrains the lateral width (w1) accordingly.
        """
        num_pts = len(self.pt_frame["s"])
        
        # Initialize with max width
        lb_w1 = np.full(num_pts, -self.MAX_LATERAL_WIDTH)
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)

        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]
            frame_t = self.pt_frame["t"][i]
            n1 = self.pt_frame["n1"][i]
            
            # Project to 2D (xy plane) for processing
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            n1_2d = np.array([n1[0], n1[1], 0.0])

            if np.linalg.norm(t_2d) < 1e-3:
                continue
            t_2d /= np.linalg.norm(t_2d)

            # 1. Contract bounds near Gates
            for gate_pos in self.gates_pos:
                gate_pos_2d = np.array([gate_pos[0], gate_pos[1], 0.0])
                r_vec_2d = gate_pos_2d - pos_2d
                
                d_abs = np.linalg.norm(r_vec_2d)
                d_long = np.dot(r_vec_2d, t_2d) # Longitudinal dist

                # Check if within influence range of gate
                if abs(d_long) <= self.GATE_CONTRACTION_LEN and abs(d_abs) <= self.GATE_CONTRACTION_LEN:
                    new_bound = self.gate_size / 2.0 - 0.1 # Shrink to fit gate
                    ub_w1[i] = min(ub_w1[i], new_bound)
                    lb_w1[i] = max(lb_w1[i], -new_bound)

            # 2. Avoid Obstacles
            for obs in self.obstacles_pos:
                obs_2d = np.array([obs[0], obs[1], 0.0])
                r_vec_2d = obs_2d - pos_2d
                
                d_abs = np.linalg.norm(r_vec_2d)
                d_long = np.dot(r_vec_2d, t_2d)

                if abs(d_long) > self.CONTRACTION_LEN or abs(d_abs) > self.CONTRACTION_LEN:
                    continue

                # Lateral distance to obstacle along normal
                w1_obs = np.dot(r_vec_2d, n1_2d)

                # Skip if obstacle is already far outside lateral width
                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS):
                    continue

                # Decision Logic: Pass Left or Right?
                if abs(w1_obs) < 0.1: # Ambiguous / Center case
                    # Determine which side offers more space
                    prop_ub = w1_obs - self.SAFETY_RADIUS
                    prop_lb = w1_obs + self.SAFETY_RADIUS
                    
                    space_on_left = prop_ub - lb_w1[i]
                    space_on_right = ub_w1[i] - prop_lb
                    
                    if space_on_left > space_on_right:
                        # Pass Left (Obstacle is right wall)
                        ub_w1[i] = min(ub_w1[i], prop_ub)
                    else:
                        # Pass Right (Obstacle is left wall)
                        lb_w1[i] = max(lb_w1[i], prop_lb)
                else:
                    # Clearly on one side
                    if w1_obs >= 0: 
                        # Obs is to the Left (positive n1), so we must be Right of it -> constrain Upper Bound
                        ub_w1[i] = min(ub_w1[i], w1_obs - self.SAFETY_RADIUS)
                    else:
                        # Obs is to the Right, constrain Lower Bound
                        lb_w1[i] = max(lb_w1[i], w1_obs + self.SAFETY_RADIUS)

        # Safety Check: Did we collapse the corridor?
        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2.0
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05
            print(f"[Geometry] Warning: Corridor collapsed at {np.sum(collapsed)} points. Forced min width.")

        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    # =========================================================================
    # Section 4: Helpers & State Retrieval
    # =========================================================================

    def angle_between_vectors(self, v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
        """Calculate the angle in radians between two vectors."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.arccos(dot_product)

    def _add_debug_statement(self, **kwargs):
        """Stores debug info for analysis."""
        self.debug_dicts.append(kwargs)

    def get_frame(self, s_query: float) -> Dict[str, NDArray]:
        """Retrieve the frame properties (pos, t, n1, etc) at a specific arc length s."""
        idx = np.searchsorted(self.pt_frame["s"], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        # Exclude 's' array from return dict for cleaner output
        return {k: self.pt_frame[k][idx] for k in self.pt_frame if k != "s"}

    def get_closest_s(self, pos_query: NDArray[np.float64], s_guess: float = 0.0, window: float = 5.0) -> float:
        """
        Finds the arc length 's' corresponding to the closest point on the path to pos_query.
        Optimized by searching only within a window around s_guess.
        """
        mask = (self.pt_frame["s"] >= s_guess - 1.0) & (self.pt_frame["s"] <= s_guess + window)
        
        if not np.any(mask):
            candidates_pos = self.pt_frame["pos"]
            candidates_s = self.pt_frame["s"]
        else:
            candidates_pos = self.pt_frame["pos"][mask]
            candidates_s = self.pt_frame["s"][mask]
            
        dists = np.linalg.norm(candidates_pos - pos_query, axis=1)
        return candidates_s[np.argmin(dists)]
    
    def get_static_bounds(self, s_query: float) -> Tuple[float, float]:
        """Lookup pre-computed corridor bounds for a given s."""
        idx = np.searchsorted(self.pt_frame["s"], s_query)
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return self.corridor_map["lb_w1"][idx], self.corridor_map["ub_w1"][idx]

    # =========================================================================
    # Section 5: Visualization
    # =========================================================================

    def plot(self):
        """Visualizes Path, Gates, Obstacles, and Corridor using Plotly."""
        fig = go.Figure()

        # 1. Flight Path
        path = self.pt_frame["pos"]
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2],
                mode="lines", line=dict(color="black", width=4), name="Centerline"
            )
        )

        # 2. Gates (Visualized as frames)
        gate_x, gate_y_list, gate_z_list = [], [], []
        hw = hh = self.gate_size / 2.0

        for i in range(len(self.gates_pos)):
            center = self.gates_pos[i]
            y_vec = self.gates_y[i]
            z_vec = self.gates_z[i]

            # Corners
            c1 = center + (y_vec * hw) + (z_vec * hh)
            c2 = center - (y_vec * hw) + (z_vec * hh)
            c3 = center - (y_vec * hw) - (z_vec * hh)
            c4 = center + (y_vec * hw) - (z_vec * hh)

            for p in [c1, c2, c3, c4, c1]:
                gate_x.append(p[0])
                gate_y_list.append(p[1])
                gate_z_list.append(p[2])
            
            # Separator
            gate_x.append(None)
            gate_y_list.append(None)
            gate_z_list.append(None)

        fig.add_trace(
            go.Scatter3d(
                x=gate_x, y=gate_y_list, z=gate_z_list,
                mode="lines", line=dict(color="blue", width=5), name="Gates"
            )
        )

        # 3. Obstacles (Cylinders)
        if len(self.obstacles_pos) > 0:
            u = np.linspace(0, 2 * np.pi, 15)
            z_pole = np.linspace(0, self.POLE_HEIGHT, 2)
            U, Z_pole = np.meshgrid(u, z_pole)

            for i, obs in enumerate(self.obstacles_pos):
                X_pole = self.SAFETY_RADIUS * np.cos(U) + obs[0]
                Y_pole = self.SAFETY_RADIUS * np.sin(U) + obs[1]
                fig.add_trace(
                    go.Surface(
                        x=X_pole, y=Y_pole, z=Z_pole,
                        colorscale=[[0, "red"], [1, "red"]],
                        opacity=0.6, showscale=False,
                        name="Obstacle" if i == 0 else None
                    )
                )

        # 4. Corridor Bounds (Visualized as walls)
        step = 10 # Downsample for plotting performance
        p_vis = path[::step]
        n1_vis = self.pt_frame["n1"][::step]
        idx = np.arange(0, len(path), step)
        
        lb = self.corridor_map["lb_w1"][idx]
        ub = self.corridor_map["ub_w1"][idx]

        wall_left = p_vis + (n1_vis * ub[:, np.newaxis])
        wall_right = p_vis + (n1_vis * lb[:, np.newaxis])

        for wall, name in [(wall_left, "Bound L"), (wall_right, "Bound R")]:
            fig.add_trace(go.Scatter3d(
                x=wall[:, 0], y=wall[:, 1], z=wall[:, 2],
                mode="lines", line=dict(color="red", width=2), name=name
            ))

        fig.update_layout(
            title="Flight Corridor & Constraints",
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()