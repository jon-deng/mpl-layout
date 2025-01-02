"""
Utilities for visualizing primitives and constraints
"""

from typing import Callable, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np

from .containers import iter_flat
from . import primitives as pr
from . import constraints as cr
from . import constructions as cn

## Functions for plotting geometric primitives


def plot_point(ax: Axes, point: pr.Point, label: Optional[str]=None, **kwargs):
    """
    Plot a point

    Parameters
    ----------
    ax: Axes
        The axes to plot in
    point: pr.Point
        The point to plot
    label: Optional[str]
        A label
    **kwargs
        Additional keyword arguments for plotting
    """
    x, y = point.value
    line, = ax.plot([x], [y], marker=".", **kwargs)
    ax.annotate(label, (x, y), ha='center', **kwargs)

def rotation_from_line(line: pr.Line) -> float:
    """
    Return the rotation of a line vector
    """
    line_vec = cn.LineVector.assem((line,))
    unit_vec = line_vec / np.linalg.norm(line_vec)

    # Since `unit_vec` has unit length, the x-component is the cosine
    theta = 180/np.pi * np.arccos(unit_vec[0])
    if unit_vec[1] < 0:
        theta = theta + 180

    return theta

def plot_line(ax: Axes, line: pr.Line, label: Optional[str]=None, **kwargs):
    """
    Plot a line

    Parameters
    ----------
    ax: Axes
        The axes to plot in
    line: pr.Line
        The line to plot
    label: Optional[str]
        A label
    **kwargs
        Additional keyword arguments for plotting
    """
    xs = np.array([point.value[0] for point in line.values()])
    ys = np.array([point.value[1] for point in line.values()])
    ax.plot(xs, ys, **kwargs)

    xmid = 1/2*xs.sum()
    ymid = 1/2*ys.sum()
    theta = rotation_from_line(line)
    ax.annotate(label, (xmid, ymid), ha='center', va='baseline', rotation=theta, **kwargs)

def plot_polygon(ax: Axes, polygon: pr.Polygon, label: Optional[str]=None, **kwargs):
    """
    Plot a `Polygon`

    Parameters
    ----------
    ax: Axes
        The axes to plot in
    polygon: pr.Polygon
        The polygon to plot
    label: Optional[str]
        A label
    **kwargs
        Additional keyword arguments for plotting
    """
    origin = polygon[f"Line0"]["Point0"].value
    points = [
        polygon[f"Line{ii}"]["Point0"] for ii in range(len(polygon))
    ]
    verts = np.array([point.value for point in points])
    poly_patch = Polygon(verts, closed=True, **kwargs)
    ax.add_patch(poly_patch)

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

def plot_generic_prim(ax: Axes, prim: pr.Primitive, label: Optional[str]=None, **kwargs):
    pass

def plot_prim(ax: Axes, prim: pr.Primitive, label: Optional[str]=None, **kwargs):
    """
    Recursively plot all child primitives of a generic primitive

    Parameters
    ----------
    ax: Axes
        The axes to plot in
    prim: pr.Primitive
        The primitive to plot
    label: Optional[str]
        A label
    **kwargs
        Additional keyword arguments for plotting
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
    prim: pr.Primitive,
) -> Callable[[Axes, tuple[pr.Primitive, ...]], None]:
    """
    Return a function that can plot a `pr.Primitive` object

    Parameters
    ----------
    prim: pr.Primitive
        The primitive to plot

    Returns
    -------
    Callable[[Axes, tuple[pr.Primitive, ...]], None]
        A function that can plot the primitive

        This function is one of the above `plot_...` function
        (see `plot_point`, `plot_line`, etc.).
    """
    if isinstance(prim, pr.Point):
        return plot_point
    elif isinstance(prim, pr.Line):
        return plot_line
    elif isinstance(prim, pr.Polygon):
        return plot_polygon
    else:
        return plot_generic_prim


def plot_prims(ax: Axes, root_prim: pr.Primitive, cmap: Colormap=mpl.colormaps['viridis']):
    """
    Plot all child primitives in a root primitive

    Parameters
    ----------
    ax: Axes
        The axes to plot in
    root_prim: pr.Primitive
        The primitive to plot
    """
    num_prims = len(root_prim)
    for ii, (label, prim) in enumerate(root_prim.items()):
        color = cmap(ii / num_prims)
        plot_prim(ax, prim, label=label, color=color)


def figure_prims(
    root_prim: pr.Primitive,
    fig_size: tuple[float, float] = (8, 8),
    major_tick_interval: float = 1.0,
    minor_tick_interval: float = 1/8
) -> tuple[Figure, Axes]:
    """
    Return a figure of a primitive

    Parameters
    ----------
    root_prim: pr.Primitive
        The primitive to plot
    fig_size: tuple[float, float]
        The figure size
    major_tick_interval, minor_tick_interval: float, float
        Major and minor tick intervals for grid lines

        By default these are 1 and 1/8 which is nice for inch dimensions.

    Returns
    -------
    fig: Figure
        The figure
    ax: Axes
        The axes
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
