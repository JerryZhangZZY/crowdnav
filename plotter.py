import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('macosx')

X_SCALE = 10
Y_SCALE = -10

class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def plot_trajectory(self, obs_length, predicted_trajectories):
        self.ax.clear()

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Trajectory Prediction')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-10, 10)
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
