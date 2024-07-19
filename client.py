import cv2
import time
import math
import socket
import numpy as np
import threading

from scipy.optimize import minimize, Bounds

from plotter import TrajectoryPlotter
from predictor import Predictor

ROBOT_ID = 0
# FINAL_TARGET = (0, 0)
FINAL_TARGET = (-10, 7)

ARUCO_TYPE = cv2.aruco.DICT_4X4_100
CALIBRATION_MAT_PATH = "calibration_matrix.npy"
DISTORTION_CO_PATH = "distortion_coefficients.npy"

OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 139
X_SCALE = 25
Y_SCALE = 25

HORIZON_LENGTH = PRED_LENGTH
# HORIZON_LENGTH = 3
NMPC_TIMESTEP = 0.3
ROBOT_RADIUS = 3
VMAX = 5
VMIN = 1
Qc = 5.0
kappa = 4.0

# FORWARD_P_GAIN = 20
# FORWARD_D_GAIN = 15
# LATERAL_P_GAIN = 20
# LATERAL_D_GAIN = 15
ROTATION_P_GAIN = 5
ROTATION_D_GAIN = 7

HEADLESS_MODE = False

upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2

# PD controller state
previous_lateral_error = 0
previous_forward_error = 0
previous_rotation_error = 0


def get_marker_pos(frame, robot_id, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """
    Get position of all pedestrians and pos+ori of the robot in frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

    pedestrains_pos = {}
    robot_pos_and_ori = None
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,

                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)

            x = tvec[0][0][0] * 100
            y = tvec[0][0][1] * -100

            if ids[i] == robot_id:
                # Convert the rotation vector to a rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec[0][0])

                # Get the orientation in degrees around the z-axis
                sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                singular = sy < 1e-6

                if not singular:
                    z_rotation = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    z_rotation = math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])

                # Convert radians to degrees
                orientation = math.degrees(z_rotation)
                robot_pos_and_ori = (x, y, orientation)
            else:
                pedestrains_pos[ids[i][0]] = (x, y)
    return pedestrains_pos, robot_pos_and_ori


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
    if norm < 0.3:
        goal = start
    else:
        dir_vec = dir_vec / norm
        start = start + dir_vec * VMIN * timestep
    return np.linspace(start, goal, number_of_steps).reshape((2 * number_of_steps))


# def compute_xref(start, goal, number_of_steps, timestep):
#     """
#     Calculate reference points with specified constraints
#     """
#     dir_vec = goal - start
#     total_distance = np.linalg.norm(dir_vec)
#
#     max_step_length = VMAX * timestep
#     min_step_length = VMIN * timestep
#
#     if total_distance >= max_step_length * (number_of_steps - 1):
#         step_length = max_step_length
#     elif total_distance >= min_step_length * (number_of_steps - 1):
#         step_length = total_distance / (number_of_steps - 1)
#     else:
#         step_length = min_step_length
#
#     dir_vec_normalized = dir_vec / total_distance if total_distance != 0 else dir_vec
#     step_points = []
#
#     for i in range(number_of_steps):
#         if i == 0:
#             next_point = start + dir_vec_normalized * step_length
#         else:
#             next_point = step_points[-1] + dir_vec_normalized * step_length
#
#         if np.linalg.norm(next_point - start) > total_distance:
#             step_points.append(goal)
#             break
#         step_points.append(next_point)
#
#     while len(step_points) < number_of_steps:
#         step_points.append(goal)
#
#     return np.array(step_points).reshape((2 * number_of_steps))





# xref = compute_xref((10,0), np.array((0,0)), HORIZON_LENGTH, NMPC_TIMESTEP)
# print(xref)
# time.sleep(1000)


def total_cost(u, robot_state, obstacle_predictions, xref):
    """
    Calculate total cost
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, obstacle_predictions) * 10
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


# # Function to convert target coordinates to the robot's coordinate system
# def transform_to_robot_coords(x_target, y_target, x_current, y_current, orientation):
#     # Translate target relative to the current position
#     delta_x = x_target - x_current
#     delta_y = y_target - y_current
#
#     # Rotate the target into the robot's coordinate system
#     angle_rad = np.deg2rad(orientation)
#     target_robot_x = delta_x * np.cos(angle_rad) - delta_y * np.sin(angle_rad)
#     target_robot_y = - delta_x * np.sin(angle_rad) - delta_y * np.cos(angle_rad)
#
#     return target_robot_x, target_robot_y


def apply_deadzone(value, deadzone):
    """
    Function to apply motor deadzone
    """
    if abs(value) < deadzone:
        return 0
    return value


def transfer_coords(target_speed, orientation):
    """
    Function to convert target speed vector to the robot's coordinate system
    """
    target_speed_x, target_speed_y = target_speed

    """Rotate the target into the robot's coordinate system"""
    angle_rad = np.deg2rad(orientation)
    robot_speed_x = target_speed_x * np.cos(angle_rad) - target_speed_y * np.sin(angle_rad)
    robot_speed_y = - target_speed_x * np.sin(angle_rad) - target_speed_y * np.cos(angle_rad)

    return robot_speed_x, robot_speed_y


def go(orientation, target_speed, max_speed, max_control):
    """
    control robot to target speed
    """
    robot_speed_x, robot_speed_y = transfer_coords(target_speed, orientation)
    k = max_control / max_speed
    lateral_control = -min(max_control, max(-max_control, int(k * robot_speed_x)))
    vertical_control = -min(max_control, max(-max_control, int(k * robot_speed_y)))
    # deadzone = 10  # Example deadzone value, you can adjust it
    # lateral_control = apply_deadzone(lateral_control, deadzone)
    # vertical_control = apply_deadzone(vertical_control, deadzone)

    rotation_control = 0
    if not HEADLESS_MODE:
        if abs(orientation) > 5:
            global previous_rotation_error
            rotation_error = -orientation
            rotation_d_error = rotation_error - previous_rotation_error
            previous_rotation_error = rotation_error
            rotation_control = -min(max_control, max(-max_control, int(ROTATION_P_GAIN * rotation_error + ROTATION_D_GAIN * rotation_d_error)))

    send_control(vertical_control, lateral_control, rotation_control)


# def goto(current, target):
#     global previous_lateral_error, previous_forward_error, previous_rotation_error
#
#     x_current = current[0]
#     y_current = current[1]
#     orientation = current[2]
#
#     x_target = target[0]
#     y_target = target[1]
#
#     # Transform target coordinates to the robot's coordinate system
#     target_robot_x, target_robot_y = transform_to_robot_coords(x_target, y_target, x_current, y_current, orientation)
#
#     # Calculate proportional (P) errors
#     lateral_error = target_robot_x
#     forward_error = target_robot_y
#
#     # Calculate derivative (D) errors
#     lateral_d_error = lateral_error - previous_lateral_error
#     forward_d_error = forward_error - previous_forward_error
#
#     # Update previous errors
#     previous_lateral_error = lateral_error
#     previous_forward_error = forward_error
#
#     # Calculate control signals using PD control
#     lateral_speed = -min(127, max(-127, int(LATERAL_P_GAIN * lateral_error + LATERAL_D_GAIN * lateral_d_error)))
#     forward_speed = -min(127, max(-127, int(FORWARD_P_GAIN * forward_error + FORWARD_D_GAIN * forward_d_error)))
#
#     # Apply deadzone
#     deadzone = 10  # Example deadzone value, you can adjust it
#     lateral_speed = apply_deadzone(lateral_speed, deadzone)
#     forward_speed = apply_deadzone(forward_speed, deadzone)
#
#     rotation_speed = 0
#
#     if not HEADLESS_MODE:
#         rotation_error = orientation
#         rotation_d_error = rotation_error - previous_rotation_error
#         previous_rotation_error = rotation_error
#         rotation_speed = min(127, max(-127, int(ROTATION_P_GAIN * rotation_error + ROTATION_D_GAIN * rotation_d_error)))
#         rotation_speed = apply_deadzone(rotation_speed, deadzone)
#
#     send_control(forward_speed, lateral_speed, rotation_speed)


def send_control(vertical, lateral, rotation):
    message = f"{vertical},{lateral},{rotation}"
    try:
        client_socket.sendall(message.encode('utf-8'))
    except BrokenPipeError:
        print("Broken pipe error: The connection was closed by the server.")
    except Exception as e:
        print(f"An error occurred: {e}")


# def get_robot_pos_and_ori(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters_create()
#
#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
#                                                                 cameraMatrix=matrix_coefficients,
#                                                                 distCoeff=distortion_coefficients)
#
#     x_current = None
#     y_current = None
#     orientation = None
#
#     # If markers are detected
#     if len(corners) > 0:
#         for i in range(len(ids)):
#             if ids[i] == ROBOT_ID:
#                 # Estimate pose of the marker with id 0
#                 rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                                distortion_coefficients)
#
#                 # Draw a square around the markers
#                 cv2.aruco.drawDetectedMarkers(frame, corners)
#
#                 # Draw Axis
#                 cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)
#
#                 # Extract position and orientation
#                 x_current = tvec[0][0][0] * 100
#                 y_current = tvec[0][0][1] * -100
#
#                 # Convert the rotation vector to a rotation matrix
#                 rotation_matrix, _ = cv2.Rodrigues(rvec[0][0])
#
#                 # Get the orientation in degrees around the z-axis
#                 sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
#                 singular = sy < 1e-6
#
#                 if not singular:
#                     z_rotation = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#                 else:
#                     z_rotation = math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])
#
#                 # Convert radians to degrees
#                 orientation = math.degrees(z_rotation)
#                 break
#
#     return x_current, y_current, orientation


def check_pos(current, target, bias):
    """
        Check if pos is at goal with bias
    """
    if target[0] + bias > current[0] > target[0] - bias and target[1] + bias > current[1] > target[1] - bias:
        return True
    else:
        return False


"""Load Social-LSTM model"""
predictor = Predictor(MODEL_NUM, EPOCH)

"""Prepare trajectory plotter"""
plotter = TrajectoryPlotter()

history_pos = {}

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.193', 12345))

k = np.load(CALIBRATION_MAT_PATH)
d = np.load(DISTORTION_CO_PATH)

video = cv2.VideoCapture(0)
width = 1920
height = 1080
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# video.set(cv2.CAP_PROP_FOCUS, 1)
time.sleep(0.5)

# video_lock = threading.Lock()

# target = FINAL_TARGET

# def control_thread():
#     while True:
#         with video_lock:
#             ret, frame = video.read()
#         if not ret:
#             break
#         with target_lock:
#             current_target = target
#         robot_pos_and_ori = get_robot_pos_and_ori(frame, ARUCO_TYPE, k, d)
#         if not any(value is None for value in robot_pos_and_ori):
#             goto(robot_pos_and_ori, current_target)
#         time.sleep(0.05)
#
#
# control_thread = threading.Thread(target=control_thread)
# control_thread.start()

while True:
    # with video_lock:
    #     ret, frame = video.read()
    ret, frame = video.read()
    if not ret:
        break
    """Get the position of all markers, including pedestrians and the robot"""
    pedestrians_pos, robot_pos_and_ori = get_marker_pos(frame, ROBOT_ID, ARUCO_TYPE, k, d)

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
                new_position = np.array([position[0], position[1] / X_SCALE, position[2] / Y_SCALE])
                frame_data.append(new_position)
        if frame_data:
            x_seq.append(np.array(frame_data))

    predicted_trajectories = None
    obstacle_predictions = []
    if x_seq:
        predicted_trajectories = predictor.predict_trajectory(x_seq, OBS_LENGTH, PRED_LENGTH, [640, 480])

        """Drawing real-time predictive trajectories"""
        # plotter.plot_trajectory(OBS_LENGTH, predicted_trajectories)

        obstacle_predictions = get_prediction_array(predicted_trajectories)

    robot_pos = None
    current_target = None

    """If robot detected"""
    if robot_pos_and_ori is not None:
        robot_pos = robot_pos_and_ori[:2]
        robot_ori = robot_pos_and_ori[2]

        """switch target"""
        if check_pos(robot_pos, FINAL_TARGET, 1):
            FINAL_TARGET = (-FINAL_TARGET[0], -FINAL_TARGET[1])
            print("switch target!")


        xref = compute_xref(robot_pos, np.array(FINAL_TARGET), HORIZON_LENGTH, NMPC_TIMESTEP)
            # vel, _ = compute_velocity(robot_pos, [], xref)
        vel, _ = compute_velocity(robot_pos, obstacle_predictions, xref)
            # target_x = robot_pos[0] + vel[0] * NMPC_TIMESTEP * 5
            # target_y = robot_pos[1] + vel[1] * NMPC_TIMESTEP * 5
            # print("NMPC")
            # with target_lock:
            #     target = (target_x, target_y)

        # print((vel[0] ** 2) + (vel[1] ** 2))
        progress = ((vel[0] ** 2) + (vel[1] ** 2)) / 25
        # Determine the number of filled-in parts of the bar
        block = int(round(50 * progress))
        # Create the bar string
        bar = "#" * block + "-" * (50 - block)

        print(f"[{bar}]")

        # print(round(vel[1], 2))

        go(robot_ori, vel, VMAX, 127)
        current_target = (robot_pos[0] + (vel[0] * NMPC_TIMESTEP), robot_pos[1] + (vel[1] * NMPC_TIMESTEP))
        # else:
        #     print("simple go")
            # with target_lock:
            #     target = FINAL_TARGET

    plotter.plot_trajectory_and_robot(OBS_LENGTH, predicted_trajectories,
                                      robot_pos,
                                      current_target)

    cv2.imshow('Camera', frame)
    cv2.waitKey(1) & 0xFF
    time.sleep(0.2)
