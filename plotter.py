import matplotlib.pyplot as plt
import matplotlib as mpl

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

    def plot_trajectory_and_robot(self, obs_length, predicted_trajectories, current, target):
        self.ax.clear()

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-1.8, 1.8)
        self.ax.set_ylim(-1.1, 1.1)

        """Plot observed points"""
        if predicted_trajectories is not None and predicted_trajectories is not None:
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
                             [observed_last[1] * Y_SCALE, predicted_first[1] * Y_SCALE], 'bo-', markersize=3,
                             linewidth=1)
                self.ax.plot(predicted_trajectories[obs_length:, i, 0].cpu().numpy() * X_SCALE,
                             predicted_trajectories[obs_length:, i, 1].cpu().numpy() * Y_SCALE,
                             'ro-', label='Predicted' if i == 0 else "",
                             markersize=3, linewidth=1)

        """Plot additional points"""
        if current is not None:
            self.ax.plot(current[0], current[1], 'mo', label='Current', markersize=10)

        if target is not None:
            self.ax.plot(target[0], target[1], 'g*', label='Target', markersize=10)

        """Plot arrow from current to target"""
        if current is not None and target is not None:
            self.ax.annotate('', xy=(target[0], target[1]),
                             xytext=(current[0], current[1]),
                             arrowprops=dict(arrowstyle="-", color='grey', linestyle='dashed', lw=1.5))

        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        plt.draw()
        plt.pause(0.001)
