"""
Utilities for a user interface/visualization of the plot layout
"""

from typing import Callable
from matplotlib.axes import Axes

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from . import primitives as pr
from . import constraints as cr

## Functions for plotting geometric primitives


def plot_point(ax: Axes, point: pr.Point, label=None, **kwargs):
    """
    Plot a `Point`
    """
    x, y = point.value
    ax.plot([x], [y], marker=".", **kwargs)


def plot_line_segment(ax: Axes, line_segment: pr.Line, label=None, **kwargs):
    """
    Plot a `LineSegment`
    """
    xs = np.array([point.value[0] for point in line_segment.children])
    ys = np.array([point.value[1] for point in line_segment.children])
    ax.plot(xs, ys, **kwargs)


def plot_polygon(ax: Axes, polygon: pr.Polygon, label=None, **kwargs):
    """
    Plot a `Polygon`
    """
    points = [polygon[f"Line0"]["Point0"]] + [
        polygon[f"Line{ii}"]["Point1"] for ii in range(len(polygon))
    ]
    xs = np.array([point.value[0] for point in points])
    ys = np.array([point.value[1] for point in points])

    (line,) = ax.plot(xs, ys, **kwargs)
    if label is not None:
        # Plot the label in the lower left corner
        ax.annotate(
            label,
            (xs[0:1], ys[0:1]),
            xycoords="data",
            xytext=(2.0, 2.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=line.get_color(),
        )


def plot_generic_prim(ax: Axes, prim: pr.Primitive, label=None, **kwargs):
    """
    Plot any child primitives of a generic primitive
    """
    for key, child_prim in prim.items():
        plot = make_plot(child_prim)
        plot(ax, child_prim, key)


## Functions for plotting arbitrary geometric primitives
def make_plot(
    prim: pr.Primitive,
) -> Callable[[Axes, tuple[pr.Primitive, ...]], None]:
    """
    Return a function that can plot a `pr.Primitive` object
    """

    if isinstance(prim, pr.Point):
        return plot_point
    elif isinstance(prim, pr.Line):
        return plot_line_segment
    elif isinstance(prim, pr.Polygon):
        return plot_polygon
    else:
        return plot_generic_prim


def plot_prims(ax: Axes, root_prim: pr.Primitive):
    """
    Plot all the child primitives in a `pr.Primitive` tree
    """

    for label, prim in root_prim.items():
        plot = make_plot(prim)
        plot(ax, prim, label=label)


def figure_prims(
    root_prim: pr.PrimitiveNode,
    fig_size: tuple[float, float] = (8, 8),
    major_tick_interval: float = 1.0,
    minor_tick_interval: float = 1/8
):
    """
    Return a figure of all primitives in a tree
    """

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(mpl.ticker.MultipleLocator(minor_tick_interval))
        axis.set_major_locator(mpl.ticker.MultipleLocator(major_tick_interval))

    ax.set_aspect(1)
    ax.grid()

    ax.set_xlabel("x [in]")
    ax.set_ylabel("y [in]")

    plot_prims(ax, root_prim)

    return (fig, ax)
