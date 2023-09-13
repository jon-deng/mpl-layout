"""
Utilities for creating `matplotlib` plot objects
"""

import typing as typ

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mpllayout import geometry as geo
from mpllayout.array import LabelledList

def subplots(
        prims: LabelledList[geo.Primitive]
    ) -> typ.Tuple[Figure, typ.Mapping[str, Axes]]:
    """
    Create `Figure` and `Axes` objects from geometric primitives

    The `Figure` and `Axes` objects are extracted based on labels in `prims`.
    A `geo.Box` primitive named 'Figure' is used to create the `Figure` with 
    corresponding dimensions.
    Any `geo.Box` primitives prefixed with 'Axes' are used to create `Axes`
    objects in the output dictionary `axs`.

    Parameters
    ----------
    prims: LabelledList[geo.Primitive]
        A list of `Primitive` objects.
        `Figure` and `Axes` objects are created from objects prefixed with
        'Figure' or 'Axes'.

    Returns
    -------
    fig, axs: typ.Tuple[Figure, typ.Mapping[str, Axes]]
        A `Figure` instance and a mapping from axes labels to `Axes` instances
        using the `Axes` object names.
    """

    width, height = width_and_height_from_box(prims['Figure'])

    fig = plt.Figure((width, height))
    axs = {
        key: fig.add_axes(rect_from_box(prim, (width, height)))
        for key, prim in prims.items()
        if 'Axes' in key and key.count('.') == 0
    }
    return fig, axs

def width_and_height_from_box(box: geo.Box) -> typ.Tuple[float, float]:
    """
    Return the width and height of a `Box`

    Parameters
    ----------
    box: geo.Box

    Returns
    -------
    typ.Tuple[float, float]
        The `(width, height)` of the box
    """

    point_bottomleft = box.prims[0]
    xmin = point_bottomleft.param[0]
    ymin = point_bottomleft.param[1]

    point_topright = box.prims[2]
    xmax = point_topright.param[0]
    ymax = point_topright.param[1]

    return (xmax-xmin), (ymax-ymin)

def rect_from_box(
        box: geo.Box, 
        fig_size: typ.Optional[typ.Tuple[float, float]]=(1, 1)
    ) -> typ.Tuple[float, float, float, float]:
    """
    Return a `rect` tuple, `(left, bottom, width, height)`, of a `Box`

    This tuple of box information can be used to create a `Bbox` object from
    `matplotlib` or an `Axes`.

    Parameters
    ----------
    box: geo.Box    
        The box
    fig_size: typ.Optional[typ.Tuple[float, float]]
        The width and height of the figure

    Returns
    -------
    xmin, ymin, width, heigth: typ.Tuple[float, float, float, float]
    """
    fig_w, fig_h = fig_size

    point_bottomleft = box.prims[0]
    xmin = point_bottomleft.param[0]/fig_w
    ymin = point_bottomleft.param[1]/fig_h

    point_topright = box.prims[2]
    xmax = point_topright.param[0]/fig_w
    ymax = point_topright.param[1]/fig_h
    width = xmax-xmin
    height = ymax-ymin

    return (xmin, ymin, width, height)
