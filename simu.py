import numpy as np
import pybullet as p
import pybullet_data
import csv

from shapely.geometry import Point, Polygon, LineString
from scipy.optimize import minimize, Bounds
from scipy.stats import chi2
from plotter import TrajectoryPlotter
from predictor import Predictor

"""Convex hull settings"""
P = 0.99

"""Environment settings"""
START_POS = [0, 0]
TARGET_POS = [0, 0]
RADIUS = 0.3
ROBOT_COLOR = [1, 0.1, 0.1, 1]
PEDESTRIAN_COLOR = [0, 0.7, 0, 1]
TRAJ_DATA_PATH = "pixel_pos.csv"

"""Social-LSTM model settings"""
OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 139
X_SCALE = 10
Y_SCALE = -10

"""NMPC settings"""
HORIZON_LENGTH = PRED_LENGTH
# HORIZON_LENGTH = 3
NMPC_TIMESTEP = 0.4
ROBOT_RADIUS = 0.3
V_MAX = 1.5
# V_MIN = 0
Qc = 0.6
kappa = 10

upper_bound = [(1 / np.sqrt(2)) * V_MAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * V_MAX] * HORIZON_LENGTH * 2


def probability_to_alpha(probability):
    df = 2
    chi2_val = chi2.ppf(probability, df)
    return np.sqrt(chi2_val)


def create_sphere(radius, color, position):
    pedestrian_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    pedestrian_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=RADIUS)
    return p.createMultiBody(baseMass=1,
                             baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=pedestrian_collision,
                             baseVisualShapeIndex=pedestrian_visual,
                             basePosition=position)


class PedestrianPool:
    """
    Dynamic management of pedestrians
    """

    def __init__(self):
        self.active_pedestrians = {}
        self.inactive_pedestrians = []

        """Create 15 pedestrians in advance and place them in the distance"""
        for _ in range(15):
            self.inactive_pedestrians.append(create_sphere(RADIUS, PEDESTRIAN_COLOR, [50, 50, 50]))

    def get_pedestrian(self, pid, position):
        if self.inactive_pedestrians:
            pedestrian_id = self.inactive_pedestrians.pop()
            p.resetBasePositionAndOrientation(pedestrian_id, position, [0, 0, 0, 1])
        else:
            pedestrian_id = create_sphere(RADIUS, PEDESTRIAN_COLOR, position)
        self.active_pedestrians[pid] = pedestrian_id
        return pedestrian_id

    def release_pedestrian(self, pid):
        pedestrian_id = self.active_pedestrians.pop(pid, None)
        if pedestrian_id:
            p.resetBasePositionAndOrientation(pedestrian_id, [50, 50, 50], [0, 0, 0, 1])
            self.inactive_pedestrians.append(pedestrian_id)


def compute_xref(start, goal, number_of_steps, timestep):
    """
    Calculate reference points
    """
    dir_vec = goal - start
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * V_MAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2 * number_of_steps))


def compute_velocity(robot_state, polygons, xref):
    """
    Calculation of control speed in x, y direction

    Final output of NMPC
    """
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost(u, robot_state, polygons, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds, tol=1e-2)
    velocity = res.x[:2]
    return velocity, res.x


def total_cost(u, robot_state, polygons, xref):
    """
    Calculate total cost
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    if polygons:
        c2 = total_collision_cost(x_robot, polygons)
        return c1 + c2
    else:
        return c1


def tracking_cost(x, xref):
    """
    Calculate tracking cost
    """
    return np.linalg.norm(x - xref)


def quickhull(points):
    def find_side(p1, p2, p):
        val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def line_dist(p1, p2, p):
        return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))

    def hull_set(p1, p2, points, side):
        ind = -1
        max_dist = 0
        for i in range(len(points)):
            temp = line_dist(p1, p2, points[i])
            if (find_side(p1, p2, points[i]) == side) and (temp > max_dist):
                ind = i
                max_dist = temp
        if ind == -1:
            hull.append(p1)
            hull.append(p2)
            return
        hull_set(points[ind], p1, points, -find_side(points[ind], p1, p2))
        hull_set(points[ind], p2, points, -find_side(points[ind], p2, p1))

    hull = []
    if len(points) < 3:
        return points.tolist()
    min_x = np.argmin(points[:, 0])
    max_x = np.argmax(points[:, 0])
    hull_set(points[min_x], points[max_x], points, 1)
    hull_set(points[min_x], points[max_x], points, -1)
    unique_hull = np.unique(hull, axis=0)
    """Sorting points to form a correct polygon"""
    center = np.mean(unique_hull, axis=0)
    sorted_hull = sorted(unique_hull, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(sorted_hull)


def sample_ellipse_points(mu_x, mu_y, sigma_x, sigma_y, corr, alpha):
    """
    Crop the 2D Gaussian distribution according to the probability and compute the coordinates of the four vertices
    of the ellipsoid shape
    """
    cov_matrix = np.array([[sigma_x ** 2, corr * sigma_x * sigma_y],
                           [corr * sigma_x * sigma_y, sigma_y ** 2]])

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    axis_lengths = alpha * np.sqrt(eigenvalues)
    ellipse_points = np.array([
        mu_x + axis_lengths[0] * eigenvectors[0, 0], mu_y + axis_lengths[0] * eigenvectors[0, 1],
        mu_x - axis_lengths[0] * eigenvectors[0, 0], mu_y - axis_lengths[0] * eigenvectors[0, 1],
        mu_x + axis_lengths[1] * eigenvectors[1, 0], mu_y + axis_lengths[1] * eigenvectors[1, 1],
        mu_x - axis_lengths[1] * eigenvectors[1, 0], mu_y - axis_lengths[1] * eigenvectors[1, 1]
    ]).reshape(4, 2)
    return ellipse_points


def calculate_hull(pred_gaussians, alpha):
    """
    Calculate convex hull for each pedestrian and return a list of polygons
    """
    polygons = []
    if pred_gaussians:
        num_pedestrians = len(pred_gaussians[0])
        for ped_index in range(num_pedestrians):
            all_points = []
            for i in range(len(pred_gaussians)):
                gaussian_params = pred_gaussians[i][ped_index]
                mux, muy, sx, sy, corr = gaussian_params
                ellipse_points = sample_ellipse_points(mux, muy, sx, sy, corr, alpha)
                all_points.extend(ellipse_points)
            all_points = np.array(all_points)
            hull_points = quickhull(all_points)
            polygons.append(Polygon(hull_points))
    return polygons


def total_collision_cost(robot, polygons):
    total_cost = 0.0
    num_pedestrians = len(pred_gaussians[0])
    for ped_index in range(num_pedestrians):
        polygon = polygons[ped_index]
        for i in range(len(pred_gaussians)):
            rob = robot[2 * i: 2 * i + 2]
            robot_point = Point(rob)
            total_cost += collision_cost_with_polygon(robot_point, polygon)
    return total_cost


def collision_cost_with_polygon(robot_point, hull_polygon):
    hull_line = LineString(hull_polygon.exterior.coords)
    nearest_point_on_hull = hull_line.interpolate(hull_line.project(robot_point))
    d = robot_point.distance(nearest_point_on_hull) - ROBOT_RADIUS
    return Qc / (1 + np.exp(kappa * (d - 2 * ROBOT_RADIUS)))


def update_state(x0, u, timestep):
    """
    Update the state of the robot
    """
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    return np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep


def scale_coords_and_gaussians(obs_pos, pred_gaussians, x_scale, y_scale):
    scaled_obs_pos = []
    for t in range(len(obs_pos)):
        scaled_coords_t = []
        for coord in obs_pos[t]:
            scaled_coords_t.append([coord[0] * x_scale, coord[1] * y_scale])
        scaled_obs_pos.append(scaled_coords_t)

    scaled_pred_gaussians = []
    for gaussians_t in pred_gaussians:
        scaled_gaussians_t = []
        for gaussian_params in gaussians_t:
            mux, muy, sx, sy, corr = gaussian_params
            scaled_gaussians_t.append([mux * x_scale, muy * y_scale, sx * x_scale, sy * y_scale, corr])
        scaled_pred_gaussians.append(scaled_gaussians_t)

    return scaled_obs_pos, scaled_pred_gaussians


def if_collided(robot_pos, pedestrian_pos, radius):
    distance = np.sqrt((robot_pos[0] - pedestrian_pos[0]) ** 2 + (robot_pos[1] - pedestrian_pos[1]) ** 2)
    return distance <= 2 * radius


"""Pybullet setup"""
p.connect(p.GUI, options="--width=768 --height=768")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89.999, cameraTargetPosition=[0, 0, 0])
p.setRealTimeSimulation(1)

"""Load robot model"""
robotId = create_sphere(RADIUS, ROBOT_COLOR, [START_POS[0], START_POS[1], RADIUS])

"""Load Social-LSTM model"""
predictor = Predictor(MODEL_NUM, EPOCH)

"""Prepare trajectory plotter"""
plotter = TrajectoryPlotter()

pedestrian_pool = PedestrianPool()
clean_data = []

"""Read dataset"""
with open(TRAJ_DATA_PATH, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    if len(data) != 4:
        raise ValueError("CSV file does not have the correct format. Expected 4 rows.")

    frames = np.array([int(frame) for frame in data[0]])
    pids = np.array([int(pid) for pid in data[1]])
    y_coords = np.array([float(y) for y in data[2]]) * Y_SCALE
    x_coords = np.array([float(x) for x in data[3]]) * X_SCALE

    clean_data = np.vstack((frames, pids, x_coords, y_coords)).T

time_steps = np.unique(clean_data[:, 0])

history_positions = {pid: [] for pid in np.unique(pids)}

collision_count = 0

"""Convert probability to alpha"""
alpha = probability_to_alpha(P)

"""Start simulation"""
for time_step in time_steps:

    robot_pos, _ = p.getBasePositionAndOrientation(robotId)

    # previous_time = time.time()
    current_data = clean_data[clean_data[:, 0] == time_step]
    current_ids = set()

    for record in current_data:
        frame, pid, x, y = int(record[0]), int(record[1]), record[2], record[3]
        current_ids.add(pid)
        new_position = np.array([x, y, RADIUS])
        if if_collided(robot_pos[:2], [x, y], RADIUS):
            collision_count += 1
            print(f"collision: {collision_count}")

        if pid not in pedestrian_pool.active_pedestrians:
            pedestrian_pool.get_pedestrian(pid, new_position)
        else:
            p.resetBasePositionAndOrientation(pedestrian_pool.active_pedestrians[pid], new_position, [0, 0, 0, 1])

    """Recycle pedestrian objects that have left"""
    for pid in list(pedestrian_pool.active_pedestrians.keys()):
        if pid not in current_ids:
            pedestrian_pool.release_pedestrian(pid)
            del history_positions[pid]

    """The ids and corresponding x, y coordinates of all active pedestrians at the current moment"""
    active_pedestrians_info = {pid: p.getBasePositionAndOrientation(pedestrian_pool.active_pedestrians[pid])[0][:2] for
                               pid in current_ids}

    """Update the historical location of each pedestrian"""
    for pid in current_ids:
        history_positions[pid].append(np.array([pid] + list(active_pedestrians_info[pid])))
        if len(history_positions[pid]) > OBS_LENGTH:
            history_positions[pid].pop(0)

    """Check if any pedestrian's historical position length satisfies OBS_LENGTH"""
    x_seq = []
    for t in range(OBS_LENGTH):
        frame_data = []
        for pid, positions in history_positions.items():
            if len(positions) == OBS_LENGTH:
                position = positions[t]
                """Zoom coordinates"""
                new_position = np.array([position[0], position[1] / X_SCALE, position[2] / Y_SCALE])
                frame_data.append(new_position)
        if frame_data:
            x_seq.append(np.array(frame_data))

    obs_pos, pred_gaussians = None, None
    if x_seq:
        obs_pos, pred_gaussians = predictor.predict_trajectory(x_seq, OBS_LENGTH, PRED_LENGTH, [640, 480])
        obs_pos, pred_gaussians = scale_coords_and_gaussians(obs_pos, pred_gaussians, X_SCALE, Y_SCALE)

    xref = compute_xref(np.array(robot_pos[:2]), TARGET_POS, HORIZON_LENGTH, NMPC_TIMESTEP)
    polygons = calculate_hull(pred_gaussians, alpha)
    vel, _ = compute_velocity(np.array(robot_pos[:2]), polygons, xref)

    plotter.plot_trajectory_and_robot(OBS_LENGTH, obs_pos, polygons, robot_pos)

    """Robot Simple Transient"""
    p.resetBasePositionAndOrientation(robotId,
                                      [vel[0] * NMPC_TIMESTEP + robot_pos[0], vel[1] * NMPC_TIMESTEP + robot_pos[1],
                                       RADIUS],
                                      [0, 0, 0, 1])
    # time_delay = NMPC_TIMESTEP - (time.time() - previous_time)
    # if time_delay > 0:
    #     time.sleep(time_delay)

"""End simulation"""
p.disconnect()
