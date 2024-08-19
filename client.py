import cv2
import time
import math
import socket
import numpy as np

from scipy.optimize import minimize, Bounds
from scipy.stats import chi2
from plotter import TrajectoryPlotter
from predictor import Predictor

"""Gaussian distribution settings"""
P = 0.1
VIP_P = {4: 0.999}
USE_GAUSSIAN = True

"""Network settings"""
ROBOT_IP = '192.168.0.193'
ROBOT_PORT = 12345

"""Camera settings"""
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

"""Environment settings"""
ROBOT_ID = 0
FINAL_TARGET = (0, 0)
SWITCH_TARGET = False
TARGET_A = (1.8, 0.8)
TARGET_B = (-1.8, -0.8)
POS_BIAS = 0.05

"""ArUco settings"""
ARUCO_TYPE = cv2.aruco.DICT_4X4_100
CALIBRATION_MAT_PATH = "calibration_matrix.npy"
DISTORTION_CO_PATH = "distortion_coefficients.npy"

"""Social-LSTM model settings"""
OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 140
SCALE = 3

"""NMPC settings"""
HORIZON_LENGTH = PRED_LENGTH
# HORIZON_LENGTH = 3
NMPC_TIMESTEP = 0.3
ROBOT_RADIUS = 0.25
V_MAX = 0.8
V_MIN = 0
Qc = 0.6
kappa = 3

"""Headless mode settings"""
HEADLESS_MODE = False
ROTATION_P_GAIN = 5
ROTATION_D_GAIN = 7

upper_bound = [(1 / np.sqrt(2)) * V_MAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * V_MAX] * HORIZON_LENGTH * 2

previous_rotation_error = 0


def probability_to_alpha(probability):
    df = 2
    chi2_val = chi2.ppf(probability, df)
    return np.sqrt(chi2_val)


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


def precompute_ellipses(pred_gaussians, alpha, x_seq, vip_alpha):
    precomputed_ellipses = []
    if pred_gaussians:
        pids = [position[0] for position in x_seq[0]]

        for ped_index in range(len(pred_gaussians[0])):
            ellipses_for_ped = []
            pid = pids[ped_index]

            if pid in vip_alpha:
                current_alpha = vip_alpha[pid]
            else:
                current_alpha = alpha

            for i in range(len(pred_gaussians)):
                gaussian_params = pred_gaussians[i][ped_index]
                mux, muy, sx, sy, corr = gaussian_params

                cov_matrix = np.array([
                    [sx**2, corr * sx * sy],
                    [corr * sx * sy, sy**2]
                ])

                eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                major_axis = current_alpha * np.sqrt(eigvals[1])
                minor_axis = current_alpha * np.sqrt(eigvals[0])
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
    u0 = np.zeros(2 * HORIZON_LENGTH)

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
    u0 = np.zeros(2 * HORIZON_LENGTH)

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

    ellipse_value = (x_transformed / major_axis)**2 + (y_transformed / minor_axis)**2

    if ellipse_value <= 1.0:
        d = 0
    else:
        x_closest = major_axis * (x_transformed / np.sqrt(x_transformed**2 + (y_transformed * major_axis / minor_axis)**2))
        y_closest = minor_axis * (y_transformed / np.sqrt((x_transformed * minor_axis / major_axis)**2 + y_transformed**2))

        dx = x_transformed - x_closest
        dy = y_transformed - y_closest
        d = np.sqrt(dx**2 + dy**2)

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


def get_marker_info(frame, robot_id, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """
    Get position of all pedestrians and pos+ori of the robot in frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

    pedestrians_pos = {}
    robot_pos_and_ori = None
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.15, matrix_coefficients,
                                                                           distortion_coefficients)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)

            x = tvec[0][0][0]
            y = -tvec[0][0][1]

            if ids[i] == robot_id:
                """Convert the rotation vector to a rotation matrix"""
                rotation_matrix, _ = cv2.Rodrigues(rvec[0][0])

                """Get the orientation in degrees around the z-axis"""
                sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                singular = sy < 1e-6

                if not singular:
                    z_rotation = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    z_rotation = math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])

                """Convert radians to degrees"""
                orientation = math.degrees(z_rotation)
                robot_pos_and_ori = (x, y, orientation)
            else:
                pedestrians_pos[ids[i][0]] = (x, y)
    return pedestrians_pos, robot_pos_and_ori


def transfer_coords(vector, orientation):
    """
    Function to convert vector from camera view to robot's coordinate system
    """
    x, y = vector
    """Rotate the target into the robot's coordinate system"""
    angle_rad = np.deg2rad(orientation)
    robot_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    robot_y = - x * np.sin(angle_rad) - y * np.cos(angle_rad)
    return robot_x, robot_y


def go(orientation, target_speed, max_speed, max_control):
    """
    control robot to target speed
    """
    robot_speed_x, robot_speed_y = transfer_coords(target_speed, orientation)
    k = max_control / max_speed
    lateral_control = -min(max_control, max(-max_control, int(k * robot_speed_x)))
    vertical_control = -min(max_control, max(-max_control, int(k * robot_speed_y)))
    rotation_control = 0
    if not HEADLESS_MODE:
        if abs(orientation) > 5:
            global previous_rotation_error
            rotation_error = -orientation
            rotation_d_error = rotation_error - previous_rotation_error
            previous_rotation_error = rotation_error
            rotation_control = -min(max_control, max(-max_control,
                                                     int(ROTATION_P_GAIN * rotation_error + ROTATION_D_GAIN * rotation_d_error)))
    send_control(vertical_control, lateral_control, rotation_control)


def send_control(vertical, lateral, rotation):
    """
    Send control command to robot
    """
    message = f"{vertical},{lateral},{rotation}"
    try:
        client_socket.sendall(message.encode('utf-8'))
    except BrokenPipeError:
        print("Broken pipe error: The connection was closed by the server.")
    except Exception as e:
        print(f"An error occurred: {e}")


def check_pos(current, target, bias):
    """
    Check if pos is at goal with bias
    """
    if target[0] + bias > current[0] > target[0] - bias and target[1] + bias > current[1] > target[1] - bias:
        return True
    else:
        return False


def scale_coords_and_gaussians(obs_pos, pred_gaussians, scale):
    scaled_obs_pos = []
    for t in range(len(obs_pos)):
        scaled_coords_t = []
        for coord in obs_pos[t]:
            scaled_coords_t.append([coord[0] * scale, coord[1] * scale])
        scaled_obs_pos.append(scaled_coords_t)

    scaled_pred_gaussians = []
    for gaussians_t in pred_gaussians:
        scaled_gaussians_t = []
        for gaussian_params in gaussians_t:
            mux, muy, sx, sy, corr = gaussian_params
            scaled_gaussians_t.append([mux * scale, muy * scale, sx * scale, sy * scale, corr])
        scaled_pred_gaussians.append(scaled_gaussians_t)

    return scaled_obs_pos, scaled_pred_gaussians


"""Connect to robot server"""
print(f"\033[34mConnecting to {ROBOT_IP}:{ROBOT_PORT}\033[0m")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ROBOT_IP, ROBOT_PORT))
print("\033[1;32mRobot connected successfully\033[0m")

"""Load Social-LSTM model"""
predictor = Predictor(MODEL_NUM, EPOCH)
print("\033[1;32mModel loaded successfully\033[0m")

"""Prepare trajectory plotter"""
plotter = TrajectoryPlotter()

history_pos = {}

"""Load calibration profile"""
print(f"\033[34mLoading calibration profile\033[0m")
k = np.load(CALIBRATION_MAT_PATH)
d = np.load(DISTORTION_CO_PATH)
print("\033[1;32mCalibration profile loaded successfully\033[0m")

"""Prepare camera"""
print(f"\033[34mSetting up camera\033[0m")
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
time.sleep(0.2)
print("\033[1;32mCamera set up successfully\033[0m")

"""Convert probability to alpha"""
alpha = probability_to_alpha(P)
vip_alpha = {}
for pid in VIP_P:
    vip_alpha[pid] = probability_to_alpha(VIP_P[pid])

while True:
    previous_time = time.time()
    ret, frame = video.read()
    if not ret:
        break
    """Get the position of all markers, including pedestrians and the robot"""
    pedestrians_pos, robot_pos_and_ori = get_marker_info(frame, ROBOT_ID, ARUCO_TYPE, k, d)

    current_ids = list(pedestrians_pos.keys())
    for pid in current_ids:
        if pid not in history_pos:
            history_pos[pid] = []

    """Update the historical location of each pedestrian"""
    for pid in current_ids:
        history_pos[pid].append(np.array([pid] + list(pedestrians_pos[pid])))
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
                new_position = np.array([position[0], position[1] / SCALE, position[2] / SCALE])
                frame_data.append(new_position)
        if frame_data:
            x_seq.append(np.array(frame_data))

    obs_pos, pred_gaussians = None, None
    obstacle_predictions = []
    if x_seq:
        obs_pos, pred_gaussians = predictor.predict_trajectory(x_seq, OBS_LENGTH, PRED_LENGTH, [640, 480])
        obs_pos, pred_gaussians = scale_coords_and_gaussians(obs_pos, pred_gaussians, SCALE)

    robot_pos, current_target = None, None

    if USE_GAUSSIAN:
        """Using gaussian distributions"""
        ellipses = precompute_ellipses(pred_gaussians, alpha, x_seq, vip_alpha)
        mean_points = None
    else:
        """Using mean points"""
        ellipses = None
        mean_points = get_mean_points(pred_gaussians)

    """If robot detected"""
    if robot_pos_and_ori is not None:
        robot_pos = robot_pos_and_ori[:2]
        robot_ori = robot_pos_and_ori[2]

        """Switch target"""
        if SWITCH_TARGET:
            if check_pos(robot_pos, FINAL_TARGET, POS_BIAS):
                if FINAL_TARGET == TARGET_A:
                    FINAL_TARGET = TARGET_B
                else:
                    FINAL_TARGET = TARGET_A
                print("Switch target!")

        """Compute reference points"""
        xref = compute_xref(robot_pos, np.array(FINAL_TARGET), HORIZON_LENGTH, NMPC_TIMESTEP)

        if USE_GAUSSIAN:
            """Using gaussian distributions"""
            vel, _ = compute_velocity(np.array(robot_pos[:2]), ellipses, xref)
        else:
            """Using mean points"""
            vel, _ = compute_velocity_using_mean_points(np.array(robot_pos[:2]), mean_points, xref)

        """Print power percentage"""
        power = ((vel[0] ** 2) + (vel[1] ** 2)) / (V_MAX ** 2)
        block = int(round(50 * power))
        bar = "#" * block + "-" * (50 - block)
        print(f"Power:[{bar}]")

        """Motion control"""
        go(robot_ori, vel, V_MAX, 127)

        """Target position at next timestep"""
        current_target = (robot_pos[0] + (vel[0] * NMPC_TIMESTEP), robot_pos[1] + (vel[1] * NMPC_TIMESTEP))

    """Drawing real-time predictive trajectories"""
    plotter.plot_trajectory_and_robot(OBS_LENGTH, obs_pos, ellipses, mean_points, robot_pos, current_target)

    cv2.imshow('Camera', frame)
    cv2.waitKey(1) & 0xFF
    time_delay = NMPC_TIMESTEP - (time.time() - previous_time)
    if time_delay > 0:
        time.sleep(time_delay)
