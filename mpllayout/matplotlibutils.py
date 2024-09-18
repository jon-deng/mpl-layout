"""
Utilities for creating `matplotlib` plot objects from primitives
"""

import typing as tp

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mpllayout import geometry as geo
from mpllayout.containers import LabelledList


def subplots(
    prim_tree: LabelledList[geo.Primitive],
) -> tp.Tuple[Figure, tp.Mapping[str, Axes]]:
    """
    Create `Figure` and `Axes` objects from geometric primitives

    The `Figure` and `Axes` objects are extracted based on labels in `prims`.
    A `geo.Box` primitive named 'Figure' is used to create the `Figure` with
    corresponding dimensions. Any `geo.Box` primitives prefixed with 'Axes' are
    used to create `Axes` objects in the output dictionary `axs`.

    Parameters
    ----------
    prims: LabelledList[geo.Primitive]
        A list of `Primitive` objects

        `Figure` and `Axes` objects are created from primitives with labels
        prefixed by 'Figure' or 'Axes'.

    Returns
    -------
    fig, axs: tp.Tuple[Figure, tp.Mapping[str, Axes]]
        A `Figure` instance and a mapping from axes labels to `Axes` instances
        using the `Axes` object names
    """

    width, height = width_and_height_from_box(prim_tree["Figure"])

    fig = plt.Figure((width, height))
    axs = {
        key: fig.add_axes(rect_from_box(prim, (width, height)))
        for key, prim in prim_tree.items()
        if "Axes" in key and key.count(".") == 0
    }
    return fig, axs


def width_and_height_from_box(box: geo.Box) -> tp.Tuple[float, float]:
    """
    Return the width and height of a `Box`

    Parameters
    ----------
    box: geo.Box

    Returns
    -------
    tp.Tuple[float, float]
        The `(width, height)` of the box
    """

    point_bottomleft = box['Line0']['Point0']
    xmin = point_bottomleft.value[0]
    ymin = point_bottomleft.value[1]

    point_topright = box['Line1']['Point1']
    xmax = point_topright.value[0]
    ymax = point_topright.value[1]

    return (xmax - xmin), (ymax - ymin)


def rect_from_box(
    box: geo.Box, fig_size: tp.Optional[tp.Tuple[float, float]] = (1, 1)
) -> tp.Tuple[float, float, float, float]:
    """
    Return a `rect` tuple, `(left, bottom, width, height)`, from a `Box`

    This tuple of box information can be used to create a `Bbox` or `Axes`
    object from `matplotlib`.

    Parameters
    ----------
    box: geo.Box
        The box
    fig_size: tp.Optional[tp.Tuple[float, float]]
        The width and height of the figure

    Returns
    -------
    xmin, ymin, width, heigth: tp.Tuple[float, float, float, float]
    """
    fig_w, fig_h = fig_size

    point_bottomleft = box['Line0/Point0']
    xmin = point_bottomleft.value[0] / fig_w
    ymin = point_bottomleft.value[1] / fig_h

    point_topright = box['Line1/Point1']
    xmax = point_topright.value[0] / fig_w
    ymax = point_topright.value[1] / fig_h
    width = xmax - xmin
    height = ymax - ymin

    return (xmin, ymin, width, height)
