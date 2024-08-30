# CrowdNav

CrowdNav is a project designed to enhance robot motion planning in environments with dense pedestrian traffic. By integrating Social-LSTM and Nonlinear Model Predictive Control (NMPC), this project offers a robust framework for predicting pedestrian trajectories and dynamically adjusting robot paths to navigate safely and efficiently in crowded spaces.

<img src="res/simu.gif" alt="simu.gif" width="400;" /> <img src="res/real.gif" alt="real.gif" width="400;" />

## Features

- **Social-LSTM Trajectory Prediction**: A deep learning model for accurately predicting pedestrian movements based on social interactions and observed behaviors.
- **Nonlinear Model Predictive Control (NMPC)**: Real-time optimization of the robot's path to avoid collisions and adhere to social norms in dynamic environments.
- **Simulation and Real-World Testing**: The project includes both simulation and real-world implementations to validate the effectiveness of the proposed framework in various scenarios.

## Project Structure

- **`main/`**: Includes code and configuration for real-world experiments.
- **`simulation/`**: Contains all simulation-related code and resources.

## Acknowledgments

Special thanks to Akin for his guidance and support, and to Valerio, Karim, and all my fellow students who assisted in the real-world experiments.
