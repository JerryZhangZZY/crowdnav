import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse

mpl.use('macosx')

X_SCALE = 2
Y_SCALE = 2


class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))

    def plot_trajectory(self, obs_length, predicted_trajectories):
        self.ax.clear()

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Trajectory Prediction')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-10, 10)

        """Plot observed points"""
        for i in range(predicted_trajectories.shape[1]):
            self.ax.plot(predicted_trajectories[:obs_length, i, 0].cpu().numpy() * X_SCALE,
                         predicted_trajectories[:obs_length, i, 1].cpu().numpy() * Y_SCALE,
                         'bo-', label='Observed' if i == 0 else "",
                         markersize=3, linewidth=1)

        """Plot predicted points"""
        for i in range(predicted_trajectories.shape[1]):
            observed_last = predicted_trajectories[obs_length - 1, i, :].cpu().numpy()
            predicted_first = predicted_trajectories[obs_length, i, :].cpu().numpy()
            self.ax.plot([observed_last[0] * X_SCALE, predicted_first[0] * X_SCALE],
                         [observed_last[1] * Y_SCALE, predicted_first[1] * Y_SCALE], 'bo-', markersize=3, linewidth=1)
            self.ax.plot(predicted_trajectories[obs_length:, i, 0].cpu().numpy() * X_SCALE,
                         predicted_trajectories[obs_length:, i, 1].cpu().numpy() * Y_SCALE,
                         'ro-', label='Predicted' if i == 0 else "",
                         markersize=3, linewidth=1)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    # def plot_trajectory_and_robot(self, obs_length, predicted_trajectories, current, target):
    #     self.ax.clear()
    #
    #     self.ax.set_xlabel('X')
    #     self.ax.set_ylabel('Y')
    #     self.ax.set_title('Observer')
    #     self.ax.set_aspect('equal', adjustable='box')
    #     self.ax.set_xlim(-2.3, 1.8)
    #     self.ax.set_ylim(-1.1, 1.2)
    #
    #     """Plot observed points"""
    #     if predicted_trajectories is not None and predicted_trajectories is not None:
    #         for i in range(predicted_trajectories.shape[1]):
    #             self.ax.plot(predicted_trajectories[:obs_length, i, 0].cpu().numpy() * X_SCALE,
    #                          predicted_trajectories[:obs_length, i, 1].cpu().numpy() * Y_SCALE,
    #                          'bo-', label='Observed' if i == 0 else "",
    #                          markersize=3, linewidth=1)
    #
    #         """Plot predicted points"""
    #         for i in range(predicted_trajectories.shape[1]):
    #             observed_last = predicted_trajectories[obs_length - 1, i, :].cpu().numpy()
    #             predicted_first = predicted_trajectories[obs_length, i, :].cpu().numpy()
    #             self.ax.plot([observed_last[0] * X_SCALE, predicted_first[0] * X_SCALE],
    #                          [observed_last[1] * Y_SCALE, predicted_first[1] * Y_SCALE], 'bo-', markersize=3,
    #                          linewidth=1)
    #             self.ax.plot(predicted_trajectories[obs_length:, i, 0].cpu().numpy() * X_SCALE,
    #                          predicted_trajectories[obs_length:, i, 1].cpu().numpy() * Y_SCALE,
    #                          'ro-', label='Predicted' if i == 0 else "",
    #                          markersize=3, linewidth=1)
    #
    #     """Plot additional points"""
    #     if current is not None:
    #         self.ax.plot(current[0], current[1], 'mo', label='Current', markersize=10)
    #
    #     if target is not None:
    #         self.ax.plot(target[0], target[1], 'g*', label='Target', markersize=10)
    #
    #     """Plot arrow from current to target"""
    #     if current is not None and target is not None:
    #         self.ax.annotate('', xy=(target[0], target[1]),
    #                          xytext=(current[0], current[1]),
    #                          arrowprops=dict(arrowstyle="-", color='grey', linestyle='dashed', lw=1.5))
    #
    #     handles, _ = self.ax.get_legend_handles_labels()
    #     if handles:
    #         self.ax.legend()
    #     plt.draw()
    #     plt.pause(0.001)

    def plot_trajectory_and_robot(self, obs_length, history_coords, pred_gaussians, current, target):
        def plot_gaussian(ax, mux, muy, sx, sy, corr):
            """ Plot 2D Gaussian as an ellipse """
            cov_matrix = np.array([[(2*sx)**2, corr * (2*sx) * (2*sy)], [corr * (2*sx) * (2*sy), (2*sy)**2]])
            vals, vecs = np.linalg.eigh(cov_matrix)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            width, height = 2 * np.sqrt(vals)
            ellipse = Ellipse(
                xy=(mux, muy),            # Center of the ellipse
                width=width,              # Width of the ellipse
                height=height,            # Height of the ellipse
                angle=angle,              # Angle of rotation
                edgecolor='r',            # Edge color
                facecolor='None',         # Face color
                linewidth=1.5             # Line width
            )
            ax.add_patch(ellipse)

        self.ax.clear()

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-2.3, 1.8)
        self.ax.set_ylim(-1.1, 1.2)

        """ Plot observed points """
        if history_coords is not None:
            num_people = len(history_coords[0])  # Number of people
            for i in range(num_people):
                person_coords = [history_coords[t][i] for t in range(obs_length)]
                person_coords = np.array(person_coords)
                self.ax.plot(person_coords[:, 0] * X_SCALE, person_coords[:, 1] * Y_SCALE,
                             'bo-', label='Observed' if i == 0 else "",
                             markersize=3, linewidth=1)

        """ Plot predicted Gaussian distributions """
        if pred_gaussians is not None:
            num_people = len(history_coords[0])  # Number of people
            num_predictions = len(pred_gaussians) // num_people

            for t in range(num_predictions):
                idx = t * num_people
                for i in range(num_people):
                    mux, muy, sx, sy, corr = pred_gaussians[idx + i]
                    plot_gaussian(self.ax, mux * X_SCALE, muy * Y_SCALE, sx * X_SCALE, sy * Y_SCALE, corr)

        """ Plot additional points """
        if current is not None:
            self.ax.plot(current[0], current[1], 'mo', label='Current', markersize=10)

        if target is not None:
            self.ax.plot(target[0], target[1], 'g*', label='Target', markersize=10)

        """ Plot arrow from current to target """
        if current is not None and target is not None:
            self.ax.annotate('', xy=(target[0], target[1]),
                             xytext=(current[0], current[1]),
                             arrowprops=dict(arrowstyle="-", color='grey', linestyle='dashed', lw=1.5))

        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        plt.draw()
        plt.pause(0.001)
