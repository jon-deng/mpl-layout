"""
Utilities for creating `matplotlib` plot objects from primitives
"""

import typing as tp

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mpllayout import geometry as geo


def subplots(
    root_prim: geo.Primitive,
) -> tp.Tuple[Figure, tp.Mapping[str, Axes]]:
    """
    Create `Figure` and `Axes` objects from geometric primitives

    The `Figure` and `Axes` objects are extracted based on labels in `root_prim`.
    A `geo.Quadrilateral` primitive named 'Figure' is used to create the `Figure` with
    corresponding dimensions. Any `geo.Quadrilateral` primitives prefixed with 'Axes' are
    used to create `Axes` objects in the output dictionary `axs`.

    Parameters
    ----------
    root_prim: geo.Primitive
        The root `Primitive` tree

        `Figure` and `Axes` objects are created from primitives with labels
        prefixed by 'Figure' or 'Axes'.

    Returns
    -------
    fig, axs: tp.Tuple[Figure, tp.Mapping[str, Axes]]
        A `Figure` instance and a mapping from axes labels to `Axes` instances
        using the `Axes` object names
    """

    width, height = width_and_height_from_quad(root_prim["Figure"])

    fig = plt.Figure((width, height))
    axs = {
        key: fig.add_axes(rect_from_box(prim, (width, height)))
        for key, prim in root_prim.items()
        if "Axes" in key and key.count(".") == 0
    }
    return fig, axs


def width_and_height_from_quad(quad: geo.Quadrilateral) -> tp.Tuple[float, float]:
    """
    Return the width and height of a quadrilateral primitive

    Parameters
    ----------
    quad: geo.Quadrilateral

    Returns
    -------
    tp.Tuple[float, float]
        The width and height of the quadrilateral
    """

    point_bottomleft = quad['Line0/Point0']
    xmin = point_bottomleft.value[0]
    ymin = point_bottomleft.value[1]

    point_topright = quad['Line1/Point1']
    xmax = point_topright.value[0]
    ymax = point_topright.value[1]

    return (xmax - xmin), (ymax - ymin)


def rect_from_box(
    quad: geo.Quadrilateral, fig_size: tp.Optional[tp.Tuple[float, float]] = (1, 1)
) -> tp.Tuple[float, float, float, float]:
    """
    Return a `rect` tuple, `(left, bottom, width, height)`, from a `geo.Quadrilateral`

    This tuple of quad information can be used to create a `Bbox` or `Axes`
    object from `matplotlib`.

    Parameters
    ----------
    quad: geo.Quadrilateral
        The quadrilateral
    fig_size: tp.Optional[tp.Tuple[float, float]]
        The width and height of the figure

    Returns
    -------
    xmin, ymin, width, heigth: tp.Tuple[float, float, float, float]
    """
    fig_w, fig_h = fig_size

    point_bottomleft = quad['Line0/Point0']
    xmin = point_bottomleft.value[0] / fig_w
    ymin = point_bottomleft.value[1] / fig_h

    point_topright = quad['Line1/Point1']
    xmax = point_topright.value[0] / fig_w
    ymax = point_topright.value[1] / fig_h
    width = xmax - xmin
    height = ymax - ymin

    return (xmin, ymin, width, height)
