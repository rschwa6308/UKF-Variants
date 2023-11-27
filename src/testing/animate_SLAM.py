import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

from probability.visualization import plot_covariance_ellipse


def draw_SLAM_system(ax, robot_trajectory, landmarks, robot_cov, landmarks_cov, robot_trajectory_gt=None, landmarks_gt=None, n_std=3):
    """
    Draw a 2D SLAM system in progress of being filtered.
     - `robot_trajectory`: `(T, 3)`     (last item is current pose)
     - `landmarks`:     `(N, 2)`
     - `robot_cov`:     `(3, 3)`
     - `landmarks_cov`: `(N, 2, 2)`
    """

    # draw robot trajectory
    robot_xs, robot_ys = robot_trajectory[:,:2].T
    ax.plot(robot_xs, robot_ys, color="tab:orange", linestyle="--", label="robot estimate")

    # draw robot pose
    robot_x, robot_y, robot_theta = robot_trajectory[-1]
    marker_rot = np.rad2deg(robot_theta) + 30
    ax.plot(robot_x, robot_y, marker=(3, 0, marker_rot), markersize=15, color="tab:orange", linestyle="--")

    # draw landmarks
    landmarks_x, landmarks_y = landmarks.T
    ax.scatter(landmarks_x, landmarks_y, marker="x", color="tab:orange", label="landmarks estimate")

    # draw robot covariance ellipse
    plot_covariance_ellipse(ax, robot_trajectory[-1,:2], robot_cov[:2,:2], facecolor="none", n_std=n_std)

    # draw landmarks covariance ellipses
    for i in range(len(landmarks)):
        plot_covariance_ellipse(ax, landmarks[i], landmarks_cov[i], facecolor="none", n_std=n_std)

    # draw ground truth robot trajectory and pose
    if robot_trajectory_gt is not None:
        # trajectory
        robot_xs_gt, robot_ys_gt = robot_trajectory_gt[:,:2].T
        ax.plot(robot_xs_gt, robot_ys_gt, color="tab:blue", label="robot gt")

        # pose
        robot_x_gt, robot_y_gt, robot_theta_gt = robot_trajectory_gt[-1]
        marker_rot = np.rad2deg(robot_theta_gt) + 30
        ax.plot(robot_x_gt, robot_y_gt, marker=(3, 0, marker_rot), markersize=15, color="tab:blue")
    
    # draw ground truth landmarks
    if landmarks_gt is not None:
        landmarks_x_gt, landmarks_y_gt = landmarks_gt.T
        ax.scatter(landmarks_x_gt, landmarks_y_gt, color="tab:blue", label="landmarks gt")
    
    ax.legend(loc="upper right")


def animate_SLAM_system(fig, ax, robot_trajectory, landmarks, robot_cov, landmarks_cov, robot_trajectory_gt=None, landmarks_gt=None, n_std=3, title=None):
    """
    Animate a 2D SLAM system being filtered.
     - `robot_trajectory`: `(T, 3)`
     - `landmarks`:     `(T, N, 2)`
     - `robot_cov`:     `(T, 3, 3)`
     - `landmarks_cov`: `(T, N, 2, 2)`
    """
    T = len(robot_trajectory)

    # pre-compute axis limits
    x_min = min(np.min(robot_trajectory[:,0]), np.min(landmarks[:,:,0]))
    x_max = max(np.max(robot_trajectory[:,0]), np.max(landmarks[:,:,0]))
    y_min = min(np.min(robot_trajectory[:,1]), np.min(landmarks[:,:,1]))
    y_max = max(np.max(robot_trajectory[:,1]), np.max(landmarks[:,:,1]))

    def ani_func(t):
        ax.clear()

        ax.set_title(title)

        ax.axis("equal")
        pad = 1.0
        ax.set_xlim(x_min-pad, x_max+pad)
        ax.set_ylim(y_min-pad, y_max+pad)

        draw_SLAM_system(ax,
            robot_trajectory[:t+1], landmarks[t],
            robot_cov[t], landmarks_cov[t],
            robot_trajectory_gt[:t+1], landmarks_gt,
            n_std=n_std
        )

    ani = animation.FuncAnimation(fig, ani_func, frames=T, interval=100, init_func=lambda: [])

    plt.close()

    return ani