import cv2
import time
import math
import socket
import numpy as np

from scipy.optimize import minimize, Bounds

from plotter import TrajectoryPlotter
from predictor import Predictor

ROBOT_ID = 0
TARGET_POS = np.array([0, 0])

ARUCO_TYPE = cv2.aruco.DICT_4X4_100
CALIBRATION_MAT_PATH = "calibration_matrix.npy"
DISTORTION_CO_PATH = "distortion_coefficients.npy"

OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 139
X_SCALE = 10
Y_SCALE = 10

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


def get_marker_pos(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

    pedestrian_pos = {}
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,

                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)

            x = tvec[0][0][0] * 100
            y = tvec[0][0][1] * 100
            pedestrian_pos[ids[i][0]] = (x, y)
    return pedestrian_pos


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


"""Load Social-LSTM model"""
predictor = Predictor(MODEL_NUM, EPOCH)

"""Prepare trajectory plotter"""
plotter = TrajectoryPlotter()

history_pos = {}

# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('192.168.0.193', 12345))

k = np.load(CALIBRATION_MAT_PATH)
d = np.load(DISTORTION_CO_PATH)

video = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# video.set(cv2.CAP_PROP_FOCUS, 1)
time.sleep(0.5)

while True:
    _, frame = video.read()
    """Get the position of all markers, including pedestrians and the robot"""
    marker_pos = get_marker_pos(frame, ARUCO_TYPE, k, d)
    """Select position of pedestrians"""
    pedestrian_pos = {id: pos for id, pos in marker_pos.items() if id != ROBOT_ID}
    cv2.imshow('Camera', frame)

    current_ids = list(pedestrian_pos.keys())
    for pid in current_ids:
        if pid not in history_pos:
            history_pos[pid] = []

    """Update the historical location of each pedestrian"""
    for pid in current_ids:
        history_pos[pid].append(np.array([pid] + list(pedestrian_pos[pid])))
        if len(history_pos[pid]) > OBS_LENGTH:
            history_pos[pid].pop(0)

    """Delete inactive pedestrian data"""
    inactive_ids = set(history_pos.keys()) - set(current_ids)
    for pid in inactive_ids:
        del history_pos[pid]

    """Check if any pedestrian's historical position length satisfies OBS_LENGTH"""
    x_seq = []
    for t in range(OBS_LENGTH):
        frame_data = []
        for pid, positions in history_pos.items():
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

    """If robot detected"""
    if 0 in marker_pos:
        robot_pos = marker_pos[ROBOT_ID]
        # print(time.time())
        xref = compute_xref(np.array(robot_pos[:2]), TARGET_POS, HORIZON_LENGTH, NMPC_TIMESTEP)
        # vel, _ = compute_velocity(np.array(robot_pos[:2]), [], xref)
        vel, _ = compute_velocity(np.array(robot_pos[:2]), obstacle_predictions, xref)
        print(vel)





    cv2.waitKey(1) & 0xFF
    time.sleep(0.5)
