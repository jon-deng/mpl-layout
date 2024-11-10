"""
Utilities for a user interface/visualization of the plot layout
"""

import typing as tp
from matplotlib.axes import Axes

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np

from . import geometry as geo
from .containers import iter_flat

## Functions for plotting geometric primitives


def plot_point(ax: Axes, point: geo.Point, label=None, **kwargs):
    """
    Plot a `Point`
    """
    x, y = point.value
    line, = ax.plot([x], [y], marker=".", **kwargs)
    ax.annotate(label, (x, y), ha='center', **kwargs)

def rotation_from_line(line: geo.Line) -> float:
    """
    Return the rotation of a line vector
    """
    line_vec = geo.line_vector(line)
    unit_vec = line_vec / np.linalg.norm(line_vec)

    # Since `unit_vec` has unit length, the x-component is the cosine
    theta = 180/np.pi * np.arccos(unit_vec[0])
    if unit_vec[1] < 0:
        theta = theta + 180

    return theta

def plot_line(ax: Axes, line: geo.Line, label=None, **kwargs):
    """
    Plot a `LineSegment`
    """
    xs = np.array([point.value[0] for point in line.children])
    ys = np.array([point.value[1] for point in line.children])
    ax.plot(xs, ys, **kwargs)

    xmid = 1/2*xs.sum()
    ymid = 1/2*ys.sum()
    theta = rotation_from_line(line)
    # ax.annotate(label, (xmid, ymid), ha='center', va='center', rotation=theta, **kwargs)
    ax.annotate(label, (xmid, ymid), ha='center', va='baseline', rotation=theta, **kwargs)

def plot_polygon(ax: Axes, polygon: geo.Polygon, label=None, **kwargs):
    """
    Plot a `Polygon`
    """
    origin = polygon[f"Line0"]["Point0"].value
    # points = [polygon[f"Line0"]["Point0"]] + [
    #     polygon[f"Line{ii}"]["Point1"] for ii in range(len(polygon))
    # ]
    # xs = np.array([point.value[0] for point in points])
    # ys = np.array([point.value[1] for point in points])

    # (line,) = ax.plot(xs, ys, **kwargs)
    if label is not None:
        # Place the label at the first point
        ax.annotate(
            label,
            (origin[0:1], origin[0:1]),
            xycoords="data",
            xytext=(2.0, 2.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            **kwargs
        )

def plot_generic_prim(ax: Axes, prim: geo.Primitive, label=None, **kwargs):
    pass

def plot_prim(ax: Axes, prim: geo.Primitive, label=None, **kwargs):
    """
    Plot all child primitives of a generic primitive
    """
    prim_height = prim.node_height()
    for child_key, child_prim in iter_flat(label, prim):
        plot = make_plot(child_prim)
        split_child_key = child_key.split("/")

        # Use the height of the child prim to increase the transparency
        # The root primitive has zero transparency while child primitives
        # have lower transparency
        depth = len(split_child_key) - 1

        # This is the height of a node relative to the tree height
        # The root node has relative height 1 and the deepest child has relative
        # height 0
        if prim_height == 0:
            s = 1
        else:
            s = (prim_height - depth)/prim_height
        alpha = 1*s + 0.2*(1-s)

        parent_key = depth*"."
        plot(ax, child_prim, label=f"{parent_key}/{split_child_key[-1]}", alpha=alpha, **kwargs)


## Functions for plotting arbitrary geometric primitives
def make_plot(
    prim: geo.Primitive,
) -> tp.Callable[[Axes, tp.Tuple[geo.Primitive, ...]], None]:
    """
    Return a function that can plot a `geo.Primitive` object
    """

    if isinstance(prim, geo.Point):
        return plot_point
    elif isinstance(prim, geo.Line):
        return plot_line
    elif isinstance(prim, geo.Polygon):
        return plot_polygon
    else:
        return plot_generic_prim


def plot_prims(ax: Axes, root_prim: geo.Primitive, cmap: Colormap=mpl.colormaps['viridis']):
    """
    Plot all the child primitives in a `geo.Primitive` tree
    """
    num_prims = len(root_prim)
    for ii, (label, prim) in enumerate(root_prim.items()):
        color = cmap(ii / num_prims)
        plot_prim(ax, prim, label=label, color=color)


def figure_prims(
    root_prim: geo.PrimitiveNode,
    fig_size: tp.Tuple[float, float] = (8, 8),
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
