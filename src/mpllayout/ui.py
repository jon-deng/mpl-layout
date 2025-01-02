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
    # Don't plot zero length lines
    if cn.Length.assem((line,)) != 0:
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
    patch_kwargs = kwargs.copy()
    patch_kwargs['alpha'] = 0.1*kwargs['alpha']
    poly_patch = Polygon(verts, closed=True, **patch_kwargs)
    ax.add_patch(poly_patch)

    # (line,) = ax.plot(xs, ys, **kwargs)
    if label is not None:
        # Place the label at the first point
        ax.annotate(
            label,
            origin,
            xycoords="data",
            xytext=(2.0, 2.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            **kwargs
        )

def plot_generic_prim(ax: Axes, prim: pr.Primitive, label: Optional[str]=None, **kwargs):
    pass

def plot_prim(
    ax: Axes,
    prim: pr.Primitive,
    prim_key: str='',
    max_label_depth: int = 99,
    **kwargs
):
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
    split_key = prim_key.split("/")

    # The primitive heigth is the maximum primitive depth for all child
    # primitives.
    prim_height = prim.node_height() + len(split_key) - 1

    # The primitive depth is how far away from the root node the primitive is.
    depth = len(split_key) - 1

    # Add labels for primitives before a maximum depth
    if depth > max_label_depth or prim_key == '':
        label = None
    else:
        parent_key = depth*"."
        label = f"{parent_key}/{split_key[-1]}"

    # Use the prim height to control opacity.
    # The root primitive has 100% opacity while deeper child primitives
    # have lower opacity.
    if prim_height == 0:
        s = 1
    else:
        s = (prim_height - depth)/prim_height
    alpha = 1*s + 0.2*(1-s)

    plot = make_plot(prim)
    plot(ax, prim, label=label, alpha=alpha, **kwargs)

    if isinstance(prim, pr.Line):
        pass
        # NOTE: Skip plotting point belonging to lines because the labels overlap
        # TODO: Implement nice plots of points in lines
        # Should try to avoid label overlap
    else:
        for child_key, child_prim in prim.items():
            plot_prim(
                ax,
                child_prim,
                prim_key=f'{prim_key}/{child_key}',
                max_label_depth=max_label_depth,
                **kwargs
            )


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
    for ii, (key, prim) in enumerate(root_prim.items()):
        color = cmap(ii / num_prims)
        plot_prim(ax, prim, prim_key=key, color=color)


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
