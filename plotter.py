import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

mpl.use('macosx')


class TrajectoryPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))

    def plot_trajectory_and_robot(self, obs_length, history_coords, polygons, current, target):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Observer')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-2.3, 2.3)
        self.ax.set_ylim(-1.3, 1.3)

        """draw history"""
        if history_coords:
            num_people = len(history_coords[0])
            for i in range(num_people):
                person_coords = [history_coords[t][i] for t in range(obs_length)]
                person_coords = np.array(person_coords)
                self.ax.plot(person_coords[:, 0], person_coords[:, 1],
                             'bo-', label='Observed' if i == 0 else "",
                             markersize=3, linewidth=1)

        if current:
            self.ax.plot(current[0], current[1], 'mo', label='Current', markersize=10)

        if target:
            self.ax.plot(target[0], target[1], 'g*', label='Target', markersize=10)

        if current and target:
            self.ax.annotate('', xy=(target[0], target[1]),
                             xytext=(current[0], current[1]),
                             arrowprops=dict(arrowstyle="-", color='grey', linestyle='dashed', lw=1.5))

        if polygons:
            for polygon in polygons:
                x, y = polygon.exterior.xy
                self.ax.plot(x, y, label='Convex Hull', color='green')

        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        plt.draw()
        plt.pause(0.001)