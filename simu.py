import numpy as np
import pybullet as p
import pybullet_data
import csv

from scipy.optimize import minimize, Bounds
from scipy.stats import chi2
from plotter import TrajectoryPlotter
from predictor import Predictor

"""Gaussian distribution settings"""
P = 0.99
USE_GAUSSIAN = True

"""Environment settings"""
START_POS = [0, 0]
TARGET_POS = [0, 0]
RADIUS = 0.3
POS_BIAS = 0.1
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
V_MIN = 0
Qc = 6
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
    if norm < POS_BIAS:
        goal = start
    elif norm > (POS_BIAS * 5):
        dir_vec = dir_vec / norm
        goal = start + dir_vec * V_MAX * timestep * number_of_steps
    else:
        dir_vec = dir_vec / norm
        start = start + dir_vec * V_MIN * timestep
    return np.linspace(start, goal, number_of_steps).reshape((2 * number_of_steps))


def precompute_ellipses(pred_gaussians, alpha):
    precomputed_ellipses = []
    if pred_gaussians:
        for ped_index in range(len(pred_gaussians[0])):
            ellipses_for_ped = []
            for i in range(len(pred_gaussians)):
                gaussian_params = pred_gaussians[i][ped_index]
                mux, muy, sx, sy, corr = gaussian_params

                cov_matrix = np.array([
                    [sx ** 2, corr * sx * sy],
                    [corr * sx * sy, sy ** 2]
                ])

                eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                major_axis = alpha * np.sqrt(eigvals[1])
                minor_axis = alpha * np.sqrt(eigvals[0])
                rotation_angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

                ellipse_params = (mux, muy, major_axis, minor_axis, rotation_angle)
                ellipses_for_ped.append(ellipse_params)

            precomputed_ellipses.append(ellipses_for_ped)

    return precomputed_ellipses


def compute_velocity(robot_state, ellipses, xref):
    """
    Calculation of control speed in x, y direction
    Using convex hulls as prediction

    Final output of NMPC
    """
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost(u, robot_state, ellipses, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


def compute_velocity_using_mean_points(robot_state, mean_points, xref):
    """
    Calculation of control speed in x, y direction
    Using mean points as prediction

    Final output of NMPC
    """
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost_using_mean_points(u, robot_state, mean_points, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


def total_cost(u, robot_state, ellipses, xref):
    """
    Calculate total cost
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    if ellipses:
        c2 = total_collision_cost(x_robot, ellipses)
        return c1 + c2
    else:
        return c1


def total_cost_using_mean_points(u, robot_state, mean_points, xref):
    """
    Calculate total cost
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    if mean_points:
        c2 = total_collision_cost_with_mean_points(x_robot, mean_points)
        return c1 + c2
    else:
        return c1


def tracking_cost(x, xref):
    """
    Calculate tracking cost
    """
    return np.linalg.norm(x - xref)


def get_mean_points(pred_gaussians):
    """
    Extract mean points (mux, muy) for each pedestrian and return a list of lists of points
    """
    mean_points = []
    if pred_gaussians:
        num_pedestrians = len(pred_gaussians[0])
        for ped_index in range(num_pedestrians):
            pedestrian_points = []
            for i in range(len(pred_gaussians)):
                gaussian_params = pred_gaussians[i][ped_index]
                mux, muy = gaussian_params[:2]  # Extract mux and muy
                pedestrian_points.append((mux, muy))
            mean_points.append(pedestrian_points)
    return mean_points


def total_collision_cost(robot, ellipses):
    total_cost = 0.0
    num_pedestrians = len(ellipses)
    for ped_index in range(num_pedestrians):
        for i in range(len(ellipses[ped_index])):
            rob = robot[2 * i: 2 * i + 2]
            robot_point = np.array(rob)
            ellipse_params = ellipses[ped_index][i]
            total_cost += collision_cost(robot_point, ellipse_params)
    return total_cost


def total_collision_cost_with_mean_points(robot, mean_points):
    total_cost = 0.0
    num_pedestrians = len(pred_gaussians[0])
    for ped_index in range(num_pedestrians):
        pedestrian_points = mean_points[ped_index]
        for i in range(len(pred_gaussians)):
            rob = robot[2 * i: 2 * i + 2]
            robot_point = np.array(rob)
            pedestrian_mean_point = np.array(pedestrian_points[i])
            total_cost += collision_cost_with_mean_points(robot_point, pedestrian_mean_point)
    return total_cost


def collision_cost(robot_point, ellipse_params):
    x_robot, y_robot = robot_point
    mux, muy, major_axis, minor_axis, rotation_angle = ellipse_params

    cos_angle = np.cos(-rotation_angle)
    sin_angle = np.sin(-rotation_angle)

    x_transformed = cos_angle * (x_robot - mux) - sin_angle * (y_robot - muy)
    y_transformed = sin_angle * (x_robot - mux) + cos_angle * (y_robot - muy)

    ellipse_value = (x_transformed / major_axis) ** 2 + (y_transformed / minor_axis) ** 2

    if ellipse_value <= 1.0:
        d = 0
    else:
        x_closest = major_axis * (
                    x_transformed / np.sqrt(x_transformed ** 2 + (y_transformed * major_axis / minor_axis) ** 2))
        y_closest = minor_axis * (
                    y_transformed / np.sqrt((x_transformed * minor_axis / major_axis) ** 2 + y_transformed ** 2))

        dx = x_transformed - x_closest
        dy = y_transformed - y_closest
        d = np.sqrt(dx ** 2 + dy ** 2)

    return Qc / (1 + np.exp(kappa * (d - 2 * ROBOT_RADIUS)))


def collision_cost_with_mean_points(robot_point, mean_point):
    d = np.linalg.norm(robot_point - mean_point)
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

    if USE_GAUSSIAN:
        """Using gaussian distributions"""
        ellipses = precompute_ellipses(pred_gaussians, alpha)
        mean_points = None
        vel, _ = compute_velocity(np.array(robot_pos[:2]), ellipses, xref)
    else:
        """Using mean points"""
        ellipses = None
        mean_points = get_mean_points(pred_gaussians)
        vel, _ = compute_velocity_using_mean_points(np.array(robot_pos[:2]), mean_points, xref)

    plotter.plot_trajectory_and_robot(OBS_LENGTH, obs_pos, ellipses, mean_points, robot_pos)

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
