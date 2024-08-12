import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

mpl.use('macosx')


class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 8))

    def plot_trajectory_and_robot(self, obs_length, obs_pos, ellipses, mean_points, robot_pos):
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

        """Draw predicted ellipses"""
        if ellipses:
            for ellipse_params_list in ellipses:
                for ellipse_params in ellipse_params_list:
                    mux, muy, major_axis, minor_axis, rotation_angle = ellipse_params

                    angle_deg = np.degrees(rotation_angle)

                    ellipse = patches.Ellipse((mux, muy), width=2 * major_axis, height=2 * minor_axis,
                                              angle=angle_deg, edgecolor='green', alpha=0.5, facecolor='none')

                    self.ax.add_patch(ellipse)

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
