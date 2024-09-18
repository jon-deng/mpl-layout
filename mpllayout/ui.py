"""
Utilities for a user interface/visualization of the plot layout
"""

import typing as typ

import matplotlib as mpl
import numpy as np

from . import geometry as geo
from .containers import LabelledList

## Functions for plotting geometric primitives


def plot_point(ax: mpl.axes.Axes, point: geo.Point, label=None, **kwargs):
    """
    Plot a `Point` primitive to an axes
    """
    x, y = point.value
    ax.plot([x], [y], marker=".", **kwargs)


def plot_line_segment(ax: mpl.axes.Axes, line_segment: geo.Line, label=None, **kwargs):
    """
    Plot a `LineSegment` primitive in an axes
    """
    xs = np.array([point.value[0] for point in line_segment.children])
    ys = np.array([point.value[1] for point in line_segment.children])
    ax.plot(xs, ys, **kwargs)


def plot_polygon(ax: mpl.axes.Axes, polygon: geo.Polygon, label=None, **kwargs):
    """
    Plot a `ClosedPolyline` primitive in an axes
    """
    points = [polygon[f'Line0']['Point0']] + [polygon[f'Line{ii}']['Point1'] for ii in range(len(polygon))]
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


## Functions for plotting arbitrary geometric primitives
def make_plot(
    prim: geo.Primitive,
) -> typ.Callable[[mpl.axes.Axes, typ.Tuple[geo.Primitive, ...]], None]:
    """
    Return a function that can plot a `geo.Primitive` object
    """

    if isinstance(prim, geo.Point):
        return plot_point
    elif isinstance(prim, geo.Line):
        return plot_line_segment
    elif isinstance(prim, geo.Polygon):
        return plot_polygon
    else:
        raise ValueError(f"No plotting function for primitive of type {type(prim)}")


def plot_prims(ax: mpl.axes.Axes, prims: LabelledList[geo.Primitive]):
    """
    Plot a collection of `geo.Primitive` objects
    """

    for label, prim in prims.items():
        plot = make_plot(prim)
        plot(ax, prim, label=label)
