import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use('macosx')


class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 8))

    def plot_trajectory_and_robot(self, obs_length, obs_pos, polygons, robot_pos):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        """Draw observed trajectories"""
        if obs_pos:
            num_people = len(obs_pos[0])
            colors = plt.cm.get_cmap('tab10', num_people)
            for i in range(num_people):
                person_coords = [obs_pos[t][i] for t in range(obs_length)]
                person_coords = np.array(person_coords)

                color = colors(i)

                self.ax.plot(person_coords[:, 0], person_coords[:, 1],
                             color=color, marker='o', linestyle='-', label='Observed' if i == 0 else "",
                             markersize=3, linewidth=1)
                self.ax.plot(person_coords[-1, 0], person_coords[-1, 1],
                             'go', markersize=10)

        """Draw robot"""
        if robot_pos:
            self.ax.plot(robot_pos[0], robot_pos[1], 'ro', label='Robot', markersize=10)

        """Draw predicted convex hulls"""
        if polygons:
            for polygon in polygons:
                x, y = polygon.exterior.xy
                self.ax.plot(x, y, color='green')

        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        plt.draw()
        plt.pause(0.001)


    def plot_trajectory_and_robot_using_mean_points(self, obs_length, obs_pos, mean_points, robot_pos):
        """
        Plot observed trajectories, predicted mean points, and the robot's position
        """
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        """Draw observed trajectories"""
        if obs_pos:
            num_people = len(obs_pos[0])
            colors = plt.cm.get_cmap('tab10', num_people)
            for i in range(num_people):
                person_coords = [obs_pos[t][i] for t in range(obs_length)]
                person_coords = np.array(person_coords)

                color = colors(i)

                self.ax.plot(person_coords[:, 0], person_coords[:, 1],
                             color=color, marker='o', linestyle='-', label='Observed' if i == 0 else "",
                             markersize=3, linewidth=1)
                self.ax.plot(person_coords[-1, 0], person_coords[-1, 1],
                             'go', markersize=10)

        """Draw robot"""
        if robot_pos:
            self.ax.plot(robot_pos[0], robot_pos[1], 'ro', label='Robot', markersize=10)

        """Draw predicted mean points"""
        if mean_points:
            num_people = len(mean_points)
            for i in range(num_people):
                mean_coords = np.array(mean_points[i])
                self.ax.plot(mean_coords[:, 0], mean_coords[:, 1],
                             'gx-', markersize=5, linewidth=1)

        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        plt.draw()
        plt.pause(0.001)
