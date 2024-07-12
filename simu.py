import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import csv
from scipy.optimize import minimize, Bounds

from plotter import TrajectoryPlotter
from predictor import Predictor

START_POS = [0, 0, 0]
TARGET_POS = np.array([0, 0])

TRAJ_DATA_PATH = "pixel_pos.csv"
INTERPOLATION_NUM = 10
OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 139
X_SCALE = 10
Y_SCALE = -10

HORIZON_LENGTH = PRED_LENGTH
# HORIZON_LENGTH = 3
NMPC_TIMESTEP = 0.588
ROBOT_RADIUS = 0.4
VMAX = 0.9
VMIN = 0.05
Qc = 5.0
kappa = 4.0

upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2


class PedestrianPool:
    """
    Dynamic management of pedestrians
    """

    def __init__(self):
        self.active_pedestrians = {}
        self.inactive_pedestrians = []

        """Create 15 pedestrians in advance and place them in the distance"""
        for _ in range(15):
            pedestrian_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
            pedestrian_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            pedestrian_id = p.createMultiBody(baseMass=1,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=pedestrian_collision,
                                              baseVisualShapeIndex=pedestrian_visual,
                                              basePosition=[50, 50, 50])
            self.inactive_pedestrians.append(pedestrian_id)

    def get_pedestrian(self, pid, position):
        if self.inactive_pedestrians:
            pedestrian_id = self.inactive_pedestrians.pop()
            p.resetBasePositionAndOrientation(pedestrian_id, position, [0, 0, 0, 1])
        else:
            pedestrian_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
            pedestrian_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            pedestrian_id = p.createMultiBody(baseMass=1,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=pedestrian_collision,
                                              baseVisualShapeIndex=pedestrian_visual,
                                              basePosition=position)
        self.active_pedestrians[pid] = pedestrian_id
        return pedestrian_id

    def release_pedestrian(self, pid):
        pedestrian_id = self.active_pedestrians.pop(pid, None)
        if pedestrian_id:
            p.resetBasePositionAndOrientation(pedestrian_id, [50, 50, 50], [0, 0, 0, 1])
            self.inactive_pedestrians.append(pedestrian_id)


def interpolate_positions(start_pos, end_pos, num_steps):
    """
    Linear interpolation
    """
    return [start_pos + (end_pos - start_pos) * t / num_steps for t in range(1, num_steps + 1)]


def compute_velocity(robot_state, obstacle_predictions, xref):
    """
    Calculation of control speed in x, y direction

    Final output of NMPC
    """
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost(u, robot_state, obstacle_predictions, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


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
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2 * number_of_steps))


def total_cost(u, robot_state, obstacle_predictions, xref):
    """
    Calculate total cost
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, obstacle_predictions)
    return c1 + c2


def tracking_cost(x, xref):
    """
    Calculate tracking cost
    """
    return np.linalg.norm(x - xref)


def total_collision_cost(robot, obstacles):
    """
    Calculate total collision cost
    """
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            obstacle = obstacles[j]
            rob = robot[2 * i: 2 * i + 2]
            obs = obstacle[2 * i: 2 * i + 2]
            total_cost += collision_cost(rob, obs)
    return total_cost


def collision_cost(x0, x1):
    """
    Calculate collision cost
    """
    d = np.linalg.norm(x0 - x1)
    return Qc / (1 + np.exp(kappa * (d - 2 * ROBOT_RADIUS)))


def get_prediction_array(tensor):
    """
    Extract the predicted trajectories from the Social-LSTM output and convert the format
    """
    last_8_matrices = tensor[-8:]
    num_rows = last_8_matrices.shape[1]
    prediction_array = []
    for row in range(num_rows):
        concatenated_row = []
        for matrix in last_8_matrices:
            scaled_row = [matrix[row][0] * X_SCALE, matrix[row][1] * Y_SCALE]
            concatenated_row.extend(scaled_row)
        prediction_array.append(np.array(concatenated_row))
    return prediction_array


def update_state(x0, u, timestep):
    """
    Update the state of the robot
    """
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    return np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep


def goto(agent, goal_x, goal_y):
    """
    Motion control of the robot
    """
    base_pos = p.getBasePositionAndOrientation(agent)
    current_x = base_pos[0][0]
    current_y = base_pos[0][1]
    current_orientation = list(p.getEulerFromQuaternion(base_pos[1]))[2]
    goal_direction = math.atan2((goal_y - current_y), (goal_x - current_x))
    if current_orientation < 0:
        current_orientation = current_orientation + 2 * math.pi
    if goal_direction < 0:
        goal_direction = goal_direction + 2 * math.pi

    theta = goal_direction - current_orientation
    if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
        theta = theta + 2 * math.pi
    elif theta > 0 and abs(theta - 2 * math.pi) < theta:
        theta = theta - 2 * math.pi

    k_linear = 15
    k_angular = 5
    linear = k_linear * (((2 / math.pi) * abs(theta) - 1) ** 2)
    angular = k_angular * theta

    rightWheelVelocity = linear + angular
    leftWheelVelocity = linear - angular

    p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=10)
    p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=10)


"""Pybullet setup"""
p.connect(p.GUI, options="--width=768 --height=768")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89.999, cameraTargetPosition=[0, 0, 0])
p.setRealTimeSimulation(1)

"""Load robot model"""
robotId = p.loadURDF("data/turtlebot.urdf", START_POS, [0, 0, 0, 1], globalScaling=2)

"""Load Social-LSTM model"""
predictor = Predictor(MODEL_NUM, EPOCH)

"""Prepare trajectory plotter"""
plotter = TrajectoryPlotter()

pedestrian_pool = PedestrianPool()
clean_data = []
interpolated_steps = {}

"""Read dataset"""
with open(TRAJ_DATA_PATH, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    if len(data) != 4:
        raise ValueError("CSV file does not have the correct format. Expected 4 rows.")

    frames = np.array([int(frame) for frame in data[0]])
    pids = np.array([int(pid) for pid in data[1]])
    y_coords = np.array([float(y) for y in data[2]]) * -10
    x_coords = np.array([float(x) for x in data[3]]) * 10

    clean_data = np.vstack((frames, pids, x_coords, y_coords)).T

time_steps = np.unique(clean_data[:, 0])

prev_positions = {}
history_positions = {pid: [] for pid in np.unique(pids)}

"""Start simulation"""
for time_step in time_steps:
    current_data = clean_data[clean_data[:, 0] == time_step]
    current_ids = set()

    for record in current_data:
        frame, pid, x, y = int(record[0]), int(record[1]), record[2], record[3]
        current_ids.add(pid)
        new_position = np.array([x, y, 0.3])

        if pid not in pedestrian_pool.active_pedestrians:
            pedestrian_id = pedestrian_pool.get_pedestrian(pid, new_position)
            prev_positions[pid] = new_position
            interpolated_steps[pid] = [new_position] * INTERPOLATION_NUM
        else:
            start_position = prev_positions[pid]
            interpolated_positions = interpolate_positions(start_position, new_position, INTERPOLATION_NUM)
            interpolated_steps[pid] = interpolated_positions
            prev_positions[pid] = new_position

    """Update pedestrian positions"""
    for i in range(INTERPOLATION_NUM):
        for pid in current_ids:
            p.resetBasePositionAndOrientation(pedestrian_pool.active_pedestrians[pid], interpolated_steps[pid][i],
                                              [0, 0, 0, 1])
        # p.stepSimulation()
        time.sleep(0.05)

    """Recycle pedestrian objects that have left"""
    for pid in list(pedestrian_pool.active_pedestrians.keys()):
        if pid not in current_ids:
            pedestrian_pool.release_pedestrian(pid)
            del prev_positions[pid]
            del interpolated_steps[pid]
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

    obstacle_predictions = []
    if x_seq:
        predicted_trajectories = predictor.predict_trajectory(x_seq, OBS_LENGTH, PRED_LENGTH, [640, 480])

        """Drawing real-time predictive trajectories"""
        plotter.plot_trajectory(OBS_LENGTH, predicted_trajectories)

        obstacle_predictions = get_prediction_array(predicted_trajectories)

    robot_pos, robot_orientation = p.getBasePositionAndOrientation(robotId)
    # print(time.time())
    xref = compute_xref(np.array(robot_pos[:2]), TARGET_POS, HORIZON_LENGTH, NMPC_TIMESTEP)
    # vel, _ = compute_velocity(np.array(robot_pos[:2]), [], xref)
    vel, _ = compute_velocity(np.array(robot_pos[:2]), obstacle_predictions, xref)

    """Robot Motion Simulation"""
    # if abs(vel[0]) < 0.2 and abs(vel[1]) < 0.2:
    #     p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=10)
    #     p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=10)
    # else:
    #     goto(robotId, vel[0] * 5 + robot_pos[0], vel[1] * 5 + robot_pos[1])

    """Robot Simple Transient"""
    p.resetBasePositionAndOrientation(robotId,
                                      [vel[0] * NMPC_TIMESTEP + robot_pos[0], vel[1] * NMPC_TIMESTEP + robot_pos[1], 0],
                                      [0, 0, 0, 1])

"""End simulation"""
p.disconnect()
