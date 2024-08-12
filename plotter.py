import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

mpl.use('macosx')


class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))

    def plot_trajectory_and_robot(self, obs_length, obs_pos, ellipses, mean_points, current, target):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-2.3, 2.3)
        self.ax.set_ylim(-1.3, 1.3)

        """Draw observed trajectories"""
        if obs_pos:
            num_people = len(obs_pos[0])
            for i in range(num_people):
                person_coords = [obs_pos[t][i] for t in range(obs_length)]
                person_coords = np.array(person_coords)
                self.ax.plot(person_coords[:, 0], person_coords[:, 1],
                             'bo-', label='Observed' if i == 0 else "",
                             markersize=3, linewidth=1)

        """Draw robot"""
        if current:
            self.ax.plot(current[0], current[1], 'mo', label='Current', markersize=10)

        """Draw robot's next target"""
        if target:
            self.ax.plot(target[0], target[1], 'g*', label='Target', markersize=10)

        """Draw an arrow from current to target"""
        if current and target:
            self.ax.annotate('', xy=(target[0], target[1]),
                             xytext=(current[0], current[1]),
                             arrowprops=dict(arrowstyle="-", color='grey', linestyle='dashed', lw=1.5))

        """Draw predicted ellipses"""
        if ellipses:
            for ellipse_params_list in ellipses:
                for ellipse_params in ellipse_params_list:
                    mux, muy, major_axis, minor_axis, rotation_angle = ellipse_params

                    # Convert the rotation angle from radians to degrees
                    angle_deg = np.degrees(rotation_angle)

                    # Create an Ellipse patch
                    ellipse = patches.Ellipse((mux, muy), width=2 * major_axis, height=2 * minor_axis,
                                              angle=angle_deg, edgecolor='green', alpha=0.5, facecolor='none')

                    # Add the ellipse to the plot
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