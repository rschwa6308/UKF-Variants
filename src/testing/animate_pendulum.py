import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patheffects as pe
from matplotlib.patches import Arrow

import sys; sys.path.append("..")
from helpers import vec


# TODO: refactor this in terms of a draw_pendulum() function (less efficient, but this is buggy...)


# def animate_pendulum(fig, ax, timesteps, states_list, controls, L1, L2, alphas=[1.0], subsample=1):
#     T = len(timesteps[::subsample])

#     artists_list = []
#     cartesian_coords_list = []

#     # initialize each pendulum
#     for i, states in enumerate(states_list):
#         carts = states[::subsample, 0, 0]
#         theta_1s = states[::subsample, 1, 0]
#         theta_2s = states[::subsample, 2, 0]
#         forces = controls[::subsample, 0, 0]

#         alpha = alphas[i]

#         cart_marker = ax.plot(
#             0, 0, marker="s", markersize=40,
#             markeredgecolor="black", markerfacecolor="tab:gray", alpha=alpha,
#             zorder=100*i+1
#         )[0]
#         link1_line = ax.plot(
#             [0, 0], [0, 0], linewidth=10, solid_capstyle="round", color="tab:blue",
#             path_effects=[pe.Stroke(linewidth=12, foreground="black"), pe.Normal()],
#             alpha=alpha, zorder=100*i+2
#         )[0]
#         link2_line = ax.plot(
#             [0, 0], [0, 0], linewidth=10, solid_capstyle="round", color="tab:blue",
#             path_effects=[pe.Stroke(linewidth=12, foreground="black"), pe.Normal()],
#             alpha=alpha, zorder=100*i+3
#         )[0]
#         joint1_marker = ax.plot(0, 0, marker=".", markersize=4, color="black", alpha=alpha, zorder=100*i+2)[0]
#         joint2_marker = ax.plot(0, 0, marker=".", markersize=4, color="black", alpha=alpha, zorder=100*i+3)[0]


#         # pre-compute cartesian coordinates
#         theta_1s_frame = np.pi/2 - theta_1s     # convert angles to pyplot coordinate frame
#         theta_2s_frame = np.pi/2 - theta_2s
#         cart_pos = np.vstack([carts, np.zeros(T)]).T
#         link1_pos = cart_pos + L1 * np.vstack([np.cos(theta_1s_frame), np.sin(theta_1s_frame)]).T
#         link2_pos = link1_pos + L2 * np.vstack([np.cos(theta_2s_frame), np.sin(theta_2s_frame)]).T

#         artists_list.append((link1_line, link2_line, joint1_marker, joint2_marker))
#         cartesian_coords_list.append((cart_pos, link1_pos, link2_pos))
    
#     ax.axhline(0, color="dimgray", linewidth=3, zorder=0)

#     timestamp_text = ax.text(
#         0.98, 0.98, "",
#         ha="right", va="top", transform=ax.transAxes,
#         fontfamily="Lucida Console", fontsize=12
#     )

#     # set axis limits
#     L = L1 + L2
#     pad = 0.2 * L
#     x_min = np.min(cart_pos[:,0]) - L - pad
#     x_max = np.max(cart_pos[:,0]) + L + pad
#     y_min = min(0.5*x_min, -L - pad)
#     y_max = max(0.5*x_max, +L + pad)

#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)

#     def ani_func(t):
#         if t % 100 == 0: print(f"{t}/{T}")

#         for i in range(len(states_list)):
#             link1_line, link2_line, joint1_marker, joint2_marker = artists_list[i]
#             cart_pos, link1_pos, link2_pos = cartesian_coords_list[i]

#             cart_marker.set_data([cart_pos[t,0]], [cart_pos[t,1]])
#             link1_line.set_data([cart_pos[t,0], link1_pos[t,0]], [cart_pos[t,1], link1_pos[t,1]])
#             link2_line.set_data([link1_pos[t,0], link2_pos[t,0]], [link1_pos[t,1], link2_pos[t,1]])
#             joint1_marker.set_data([cart_pos[t,0]], [cart_pos[t,1]])
#             joint2_marker.set_data([link1_pos[t,0]], [link1_pos[t,1]])

#         timestamp_text.set_text(f"t = {timesteps[t*subsample]:0.1f}s")

#         for p in ax.patches: p.remove()
#         force = forces[t]
#         if abs(force) > 0:
#             arrow_len = 0.5 * np.log(abs(force))
#             ax.add_patch(Arrow(
#                 cart_pos[t,0], 2.0, np.sign(force) * arrow_len, 0,
#                 facecolor="tab:orange", edgecolor=None,
#                 width=1.5, zorder=0
#             ))
    
#     ax.set_aspect("equal")

#     sec_per_frame = (timesteps[1] - timesteps[0])*subsample
#     print(f"FPS: {1 / sec_per_frame:.0f}")
#     ani = animation.FuncAnimation(fig, ani_func, frames=T, interval=1000*sec_per_frame)

#     plt.close()

#     return ani




def draw_pendulum(ax, cart, theta_1, theta_2, L1, L2, cart_color="tab:gray", linkage_color="tab:blue", alpha=1.0, label=None, base_zorder=0):
    theta_1_frame = np.pi/2 - theta_1     # convert angles to pyplot coordinate frame
    theta_2_frame = np.pi/2 - theta_2

    cart_pos = vec(cart, 0)
    link1_pos = cart_pos + L1 * vec(np.cos(theta_1_frame), np.sin(theta_1_frame))
    link2_pos = link1_pos + L2 * vec(np.cos(theta_2_frame), np.sin(theta_2_frame))

    # draw cart
    ax.plot(
        cart_pos[0,0], 0, marker="s", markersize=35,
        markeredgecolor="black", markerfacecolor=cart_color,
        alpha=alpha, zorder=base_zorder+1
    )

    # draw linkage arms
    lw = 8
    ax.plot(
        [cart_pos[0,0], link1_pos[0,0]], [cart_pos[1,0], link1_pos[1,0]],
        linewidth=lw, solid_capstyle="round", color=linkage_color,
        path_effects=[
            pe.Stroke(linewidth=lw+2.5, foreground="black", alpha=alpha),
            pe.Stroke(linewidth=lw, foreground="white", alpha=0.2),
            pe.Normal()
        ],
        alpha=alpha, zorder=base_zorder+2,
        label=label
    )
    ax.plot(
        [link1_pos[0,0], link2_pos[0,0]], [link1_pos[1,0], link2_pos[1,0]],
        linewidth=lw, solid_capstyle="round", color=linkage_color,
        path_effects=[
            pe.Stroke(linewidth=lw+2.5, foreground="black", alpha=alpha),
            pe.Stroke(linewidth=lw, foreground="white", alpha=0.2),
            pe.Normal()
        ],
        alpha=alpha, zorder=base_zorder+3
    )

    # draw joints
    ax.plot(
        cart_pos[0,0], cart_pos[1,0],
        marker=".", markersize=4, color="black",
        alpha=alpha, zorder=base_zorder+2
    )
    ax.plot(
        link1_pos[0,0], link1_pos[1,0],
        marker=".", markersize=4, color="black",
        alpha=alpha, zorder=base_zorder+3
    )




def animate_pendulum(fig, ax, timesteps, states_list, controls, L1, L2, colors=None, alphas=None, labels=None, subsample=1):
    N = len(states_list)

    if colors is None:
        colors = [("tab:gray", "tab:blue")] * N

    if alphas is None:
        alphas = [1.0] * N
    
    if labels is None:
        labels = [None] * N

    T = len(timesteps[::subsample])

    # pre-compute fixed axis limits
    L = L1 + L2
    x_pad = 0.1 * L
    y_pad = 0.3 * L
    x_min = min(np.min(states[:,0]) for states in states_list) - L - x_pad
    x_max = max(np.max(states[:,0]) for states in states_list) + L + x_pad
    y_min = -L - y_pad
    y_max = +L + y_pad


    def ani_func(t):
        if t % 100 == 0: print(f"{t}/{T}")
        if t == T-1: print(f"{T}/{T}")

        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # draw cart rail
        ax.axhline(0, color="dimgray", linewidth=3, zorder=0)

        # draw pendulums
        for i in range(N):
            state = states_list[i][t*subsample]
            draw_pendulum(
                ax, state[0,0], state[1,0], state[2,0],
                L1, L2, *colors[i], alphas[i], base_zorder=100*i,
                label=labels[i]
            )
        
        if any(l is not None for l in labels):
            ax.legend(loc="upper left")

        # draw timestamp
        timestamp_text = ax.text(
            0.98, 0.98, "",
            ha="right", va="top", transform=ax.transAxes,
            fontfamily="Lucida Console", fontsize=12
        )
        timestamp_text.set_text(f"t = {timesteps[t*subsample]:0.1f}s")

        # draw input force arrow
        for p in ax.patches: p.remove()
        force = controls[t*subsample,0,0]
        if abs(force) > 0:
            arrow_x = states_list[0][t*subsample,0,0]
            arrow_len = 0.5 * np.log(abs(force))
            ax.add_patch(Arrow(
                arrow_x, 1.5, np.sign(force) * arrow_len, 0,
                facecolor="tab:orange", edgecolor=None,
                width=1.5, zorder=0
            ))
    
    ax.set_aspect("equal")

    sec_per_frame = (timesteps[1] - timesteps[0])*subsample
    print(f"FPS: {1 / sec_per_frame:.0f}")
    ani = animation.FuncAnimation(fig, ani_func, frames=T, interval=1000*sec_per_frame, init_func=lambda: [])

    plt.close()

    return ani
