from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch
import numpy as np

from helpers import query_pdf


def plot_pdf(ax: plt.Axes, domain: np.array, pdf_values: np.array, transpose_axis=False, **args):
    "Plot a PDF over an n-point domain represented by n-1 bins"
    dx = domain[1] - domain[0]

    if not transpose_axis:
        ax.fill_between(domain[:-1] + dx/2, pdf_values, edgecolor="black", **args)
        ax.set_ylim(0, None)

    else:
        ax.fill_betweenx(domain[:-1] + dx/2, pdf_values,edgecolor="black", **args)
        ax.set_xlim(0, None)


def plot_pdf_transform(domain, prior, transform_values, posterior, slice_highlights=[], highlight_color="tab:orange", transform_label=None, title=None):
    fig, axes = plt.subplots(
        2, 2, figsize=(6, 6),
        sharex="col", sharey="row",
        constrained_layout=True
    )
    axes[1, 0].axis("off")

    # plot PDFs
    plot_pdf(axes[1, 1], domain, prior)
    plot_pdf(axes[0, 0], domain, posterior, transpose_axis=True)

    # plot transform
    axes[0, 1].plot(domain, transform_values, label=transform_label)

    # set PDF axis limits explicitly so they match
    pmax = max(np.max(prior), np.max(posterior)) * 1.1
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

    # optional: make a legend for the transform
    if transform_label is not None:
        axes[0, 1].legend()

    # for given slices of the domain, highlight the preimage and image under the transform
    for x_low, x_high in slice_highlights:
        domain_step = domain[1] - domain[0]
        x_low_idx = int((x_low - domain[0]) / domain_step)
        x_high_idx = int((x_high - domain[0]) / domain_step)

        prior_low = query_pdf(domain, prior, x_low)
        prior_high = query_pdf(domain, prior, x_high)

        y_low = transform_values[x_low_idx]
        y_high = transform_values[x_high_idx]

        posterior_low = query_pdf(domain, posterior, y_low)
        posterior_high = query_pdf(domain, posterior, y_high)

        fig.add_artist(ConnectionPatch(
            (x_low, prior_low), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8
        ))

        fig.add_artist(ConnectionPatch(
            (x_high, prior_high), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8
        ))

        fig.add_artist(ConnectionPatch(
            (posterior_low, y_low), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8
        ))

        fig.add_artist(ConnectionPatch(
            (posterior_high, y_high), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8
        ))

        mask = (x_low <= domain) & (domain <= x_high)
        plot_pdf(axes[1, 1], domain[mask], prior[mask[:-1]][:-1], linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color)

        mask = (y_low <= domain) & (domain <= y_high)
        plot_pdf(axes[0, 0], domain[mask], posterior[mask[:-1]][:-1], transpose_axis=True, linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color)
