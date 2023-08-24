"""
Utilities for a user interface/visualization of plot layout
"""

import typing as typ

import matplotlib as mpl
import numpy as np

from . import geometry as geo
from .array import LabelledList

## Function for plotting geometric primitives

def plot_point(
        ax: mpl.axes.Axes, point: geo.Point, **kwargs
    ):
    x, y = point.param
    ax.plot([x], [y], **kwargs)

def plot_line_segment(
        ax: mpl.axes.Axes, line_segment: geo.LineSegment, **kwargs
    ):
    xs = np.array([point.param[0] for point in line_segment.prims])
    ys = np.array([point.param[1] for point in line_segment.prims])
    ax.plot(xs, ys, **kwargs)

def plot_closed_polyline(
        ax: mpl.axes.Axes, polyline: geo.ClosedPolyline, **kwargs
    ):
    closed_prims = polyline.prims[:]+(polyline.prims[0],)
    xs = np.array([point.param[0] for point in closed_prims])
    ys = np.array([point.param[1] for point in closed_prims])
    ax.plot(xs, ys, **kwargs)

def make_plot(
        prim: geo.Primitive
    ) -> typ.Callable[[mpl.axes.Axes, typ.Tuple[geo.Primitive, ...]], None]:

    if isinstance(prim, geo.Point):
        return plot_point
    elif isinstance(prim, geo.LineSegment):
        return plot_line_segment
    elif isinstance(prim, geo.ClosedPolyline):
        return plot_closed_polyline
    else:
        raise ValueError(f"No plotting function for primitive of type {type(prim)}")

def plot_prims(
        ax: mpl.axes.Axes, prims: LabelledList[geo.Primitive]
    ):

    for prim in prims:
        plot = make_plot(prim)
        plot(ax, prim)
