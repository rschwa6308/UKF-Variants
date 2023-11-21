from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch
import numpy as np

# from helpers import query_pdf


def plot_pdf_values(ax: plt.Axes, sample_points: np.array, pdf_values: np.array, transpose_axis=False, **args):
    "Plot a PDF evaluated at sample points"

    if not transpose_axis:
        ax.fill_between(sample_points, pdf_values, edgecolor="black", **args)
        ax.set_ylim(0, None)

    else:
        ax.fill_betweenx(sample_points, pdf_values,edgecolor="black", **args)
        ax.set_xlim(0, None)


def plot_pdf_transform(domain, prior, transform_values, posterior, slice_highlights=[], highlight_color="tab:orange", prior_label=None, posterior_label=None, transform_label=None, title=None):
    fig, axes = plt.subplots(
        2, 2, figsize=(6, 6),
        sharex="col", sharey="row",
        constrained_layout=True
    )
    axes[1, 0].axis("off")

    # plot PDFs
    plot_pdf_values(axes[1, 1], domain, prior, label=prior_label, zorder=2)
    plot_pdf_values(axes[0, 0], domain, posterior, transpose_axis=True, label=posterior_label, zorder=2)

    # plot transform
    axes[0, 1].plot(domain, transform_values, label=transform_label)

    # set PDF axis limits explicitly so they match
    pmax = max(np.max(prior), np.max(posterior)) * 1.2
    axes[0, 0].set_xlim(0, pmax)
    axes[1, 1].set_ylim(0, pmax)

    # set PDF axis ticks explicitly so they match
    axes[1, 1].set_xticks(np.linspace(domain[0], domain[-1], 5))
    axes[0, 0].set_yticks(np.linspace(domain[0], domain[-1], 5))

    # rotate posterior PDF tick labels
    axes[0, 0].tick_params(axis="y", rotation=-90)

    # add subplot captions
    axes[1, 1].set_xlabel("prior")
    axes[0, 0].set_ylabel("posterior", rotation=-90, va="top")

    # optional: set figure title
    if title is not None:
        fig.suptitle(title)

    # optional: make legends
    if prior_label is not None:     axes[1, 1].legend(loc="upper right")
    if posterior_label is not None: axes[0, 0].legend(loc="upper right")
    if transform_label is not None: axes[0, 1].legend(loc="upper right")
    

    # for given slices of the domain, highlight the preimage and image under the transform
    for x_low, x_high in slice_highlights:
        domain_step = domain[1] - domain[0]
        x_low_idx = int((x_low - domain[0]) / domain_step)
        x_high_idx = int((x_high - domain[0]) / domain_step)

        y_low = transform_values[x_low_idx]
        y_high = transform_values[x_high_idx]

        axes[1,1].add_artist(ConnectionPatch(
            (x_low, 0), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[1,1].add_artist(ConnectionPatch(
            (x_high, 0), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[0,0].add_artist(ConnectionPatch(
            (0, y_low), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[0,0].add_artist(ConnectionPatch(
            (0, y_high), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        mask = (x_low <= domain) & (domain <= x_high)
        plot_pdf_values(
            axes[1, 1], domain[mask], prior[mask[:-1]][:-1],
            linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color,
            zorder=2
        )

        mask = (y_low <= domain) & (domain <= y_high)
        plot_pdf_values(
            axes[0, 0], domain[mask], posterior[mask[:-1]][:-1], transpose_axis=True,
            linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color,
            zorder=2
        )

    # change subplot draw order so that the connection patches draw on top of the transform plot
    axes[0, 0].set_zorder(1)
    axes[0, 1].set_zorder(0)
    axes[1, 1].set_zorder(1)

import matplotlib.patches
import matplotlib.transforms


def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, edgecolor="black", **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional distribution
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        edgecolor=edgecolor, **kwargs
    )

    mean_x, mean_y = mean.flatten()

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
