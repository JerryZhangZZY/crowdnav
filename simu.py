import numpy as np
import pybullet as p
import pybullet_data
import time
import csv

from plotter import TrajectoryPlotter
from predictor import Predictor

TRAJ_DATA_PATH = "pixel_pos.csv"
INTERPOLATION_NUM = 10
OBS_LENGTH = 5
PRED_LENGTH = 8
MODEL_NUM = 3
EPOCH = 139
X_SCALE = 10
Y_SCALE = -10

class PedestrianPool:
    def __init__(self):
        self.active_pedestrians = {}
        self.inactive_pedestrians = []

        # 提前创建12个行人并放置到远处
        for _ in range(12):
            pedestrian_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
            pedestrian_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            pedestrian_id = p.createMultiBody(baseMass=1,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=pedestrian_collision,
                                              baseVisualShapeIndex=pedestrian_visual,
                                              basePosition=[1000, 1000, 1000])  # 放置到远处
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
            p.resetBasePositionAndOrientation(pedestrian_id, [1000, 1000, 1000], [0, 0, 0, 1])  # 放置到远处
            self.inactive_pedestrians.append(pedestrian_id)


def interpolate_positions(start_pos, end_pos, num_steps):
    # 线性插值计算中间位置
    return [start_pos + (end_pos - start_pos) * t / num_steps for t in range(1, num_steps + 1)]


# 设置 PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89.999, cameraTargetPosition=[0, 0, 0])

# 加载模型
predictor = Predictor(MODEL_NUM, EPOCH)

plotter = TrajectoryPlotter()

pedestrian_pool = PedestrianPool()
clean_data = []
interpolated_steps = {}

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

for time_step in time_steps:
    current_data = clean_data[clean_data[:, 0] == time_step]
    current_ids = set()

    # 计算当前时间步的插值位置
    for record in current_data:
        frame, pid, x, y = int(record[0]), int(record[1]), record[2], record[3]
        current_ids.add(pid)
        new_position = np.array([x, y, 0.3])

        if pid not in pedestrian_pool.active_pedestrians:  # 创建新行人
            pedestrian_id = pedestrian_pool.get_pedestrian(pid, new_position)
            prev_positions[pid] = new_position
            interpolated_steps[pid] = [new_position] * INTERPOLATION_NUM
        else:
            start_position = prev_positions[pid]
            interpolated_positions = interpolate_positions(start_position, new_position, INTERPOLATION_NUM)
            interpolated_steps[pid] = interpolated_positions
            prev_positions[pid] = new_position

    # 更新所有行人的位置
    for i in range(INTERPOLATION_NUM):
        for pid in current_ids:
            p.resetBasePositionAndOrientation(pedestrian_pool.active_pedestrians[pid], interpolated_steps[pid][i],
                                              [0, 0, 0, 1])
        # p.stepSimulation()
        time.sleep(0.001)

    # 释放不再出现的行人
    for pid in list(pedestrian_pool.active_pedestrians.keys()):
        if pid not in current_ids:
            pedestrian_pool.release_pedestrian(pid)
            del prev_positions[pid]
            del interpolated_steps[pid]
            del history_positions[pid]

    # 打印当前时刻所有active的行人的id和对应的x, y坐标
    active_pedestrians_info = {pid: p.getBasePositionAndOrientation(pedestrian_pool.active_pedestrians[pid])[0][:2] for
                               pid in current_ids}

    # 更新每个行人的历史位置
    for pid in current_ids:
        history_positions[pid].append(np.array([pid] + list(active_pedestrians_info[pid])))
        if len(history_positions[pid]) > OBS_LENGTH:
            history_positions[pid].pop(0)

    # 检查是否有行人的历史位置达到OBS_LENGTH
    x_seq = []
    for t in range(OBS_LENGTH):
        frame_data = []
        for pid, positions in history_positions.items():
            if len(positions) == OBS_LENGTH:
                position = positions[t]
                # 对坐标进行转换
                new_position = np.array([position[0], position[1] / X_SCALE, position[2] / Y_SCALE])
                frame_data.append(new_position)
        if frame_data:
            x_seq.append(np.array(frame_data))


    if x_seq:
        # print(x_seq)
        predicted_trajectories = predictor.predict_trajectory(x_seq, OBS_LENGTH, PRED_LENGTH, [640, 480])
        plotter.plot_trajectory(OBS_LENGTH, predicted_trajectories)

# 结束模拟
p.disconnect()