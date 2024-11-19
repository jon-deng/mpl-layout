"""
Utilities for creating `matplotlib` elements from geometric primitives
"""

from typing import Optional
import warnings

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import primitives as pr
from . import constraints as cr


def subplots(
    root_prim: pr.Primitive,
    fig_key: str = "Figure",
    axs_keys: Optional[list[str]] = None,
) -> tuple[Figure, dict[str, Axes]]:
    """
    Create `Figure` and `Axes` objects from geometric primitives

    The `Figure` and `Axes` objects are extracted based on labels in `root_prim`.
    A `pr.Quadrilateral` primitive named 'Figure' is used to create the `Figure` with
    corresponding dimensions. Any `pr.Quadrilateral` primitives prefixed with 'Axes' are
    used to create `Axes` objects in the output dictionary `axs`.

    Parameters
    ----------
    root_prim: pr.Primitive
        The root `Primitive` tree

        `Figure` and `Axes` objects are created from primitives with labels
        prefixed by 'Figure' or 'Axes'.

    Returns
    -------
    fig, axs: tuple[Figure, dict[str, Axes]]
        A `Figure` instance and a mapping from axes labels to `Axes` instances
        using the `Axes` object names
    """
    # Create the `Figure` instance
    fig = plt.figure(figsize=(1, 1))

    # Assume all axes are prefixed by "Axes" if there are no keys provided
    if axs_keys is None:
        axs_keys = [key for key in root_prim.keys() if "Axes" in key]

    # Create all `Axes` instances
    unit_rect = (0, 0, 1, 1)
    key_to_ax = {key: fig.add_axes(unit_rect) for key in axs_keys}

    # Update positions figures and axes
    fig, key_to_ax = update_subplots(root_prim, fig_key, fig, key_to_ax)

    return fig, key_to_ax


def update_subplots(
    root_prim: pr.Primitive,
    fig_key: str,
    fig: Figure,
    key_to_ax: dict[str, Axes],
):
    # Set Figure position
    quad = root_prim[fig_key]
    fig_size = np.array(width_and_height_from_quad(quad))
    fig.set_size_inches(fig_size)

    # Set Axes properties/position
    for key, ax in key_to_ax.items():
        # Set Axes dimensions
        quad = root_prim[f"{key}/Frame"]
        ax.set_position(rect_from_box(quad, fig_size))

        # Set x/y axis properties
        axis_prefixes = ("X", "Y")
        axis_tuple = (ax.xaxis, ax.yaxis)

        for axis_prefix, axis in zip(axis_prefixes, axis_tuple):
            # Set the axis label position
            axis_label = f"{axis_prefix}AxisLabel"
            if axis_label in root_prim[key]:
                axis_label_point: pr.Point = root_prim[f"{key}/{axis_label}"]
                label_coords = axis_label_point.value
                axis.set_label_coords(
                    *(label_coords / fig_size), transform=fig.transFigure
                )

            # Set the axis tick position
            axis_bbox = f"{axis_prefix}Axis"
            if axis_bbox in root_prim[key]:
                axis_quad = root_prim[f"{key}/{axis_bbox}"]
                axis_tick_position = find_axis_position(quad, axis_quad)
                axis.set_ticks_position(axis_tick_position)

    return fig, key_to_ax


def find_axis_position(axes_frame: pr.Quadrilateral, axis: pr.Quadrilateral):
    coincident_line = cr.CoincidentLines()
    params = {"reverse": True}
    bottom_res = coincident_line((axes_frame["Line0"], axis["Line2"]), params)
    top_res = coincident_line((axes_frame["Line2"], axis["Line0"]), params)
    left_res = coincident_line((axes_frame["Line3"], axis["Line1"]), params)
    right_res = coincident_line((axes_frame["Line1"], axis["Line3"]), params)

    residuals = tuple(
        np.linalg.norm(res) for res in (bottom_res, top_res, left_res, right_res)
    )
    residual_positions = ("bottom", "top", "left", "right")

    if not np.isclose(np.min(residuals), 0):
        warnings.warn("The axis isn't closely aligned with any of the axes sides")
    position = residual_positions[np.argmin(residuals)]
    return position


def width_and_height_from_quad(quad: pr.Quadrilateral) -> tuple[float, float]:
    """
    Return the width and height of a quadrilateral primitive

    Parameters
    ----------
    quad: pr.Quadrilateral

    Returns
    -------
    tuple[float, float]
        The width and height of the quadrilateral
    """

    point_bottomleft = quad["Line0/Point0"]
    xmin = point_bottomleft.value[0]
    ymin = point_bottomleft.value[1]

    point_topright = quad["Line1/Point1"]
    xmax = point_topright.value[0]
    ymax = point_topright.value[1]

    return (xmax - xmin), (ymax - ymin)


def rect_from_box(
    quad: pr.Quadrilateral, fig_size: Optional[tuple[float, float]] = (1, 1)
) -> tuple[float, float, float, float]:
    """
    Return a `rect` tuple, `(left, bottom, width, height)`, from a `pr.Quadrilateral`

    This tuple of quad information can be used to create a `Bbox` or `Axes`
    object from `matplotlib`.

    Parameters
    ----------
    quad: pr.Quadrilateral
        The quadrilateral
    fig_size: Optional[tuple[float, float]]
        The width and height of the figure

    Returns
    -------
    xmin, ymin, width, heigth: tuple[float, float, float, float]
    """
    fig_w, fig_h = fig_size

    point_bottomleft = quad["Line0/Point0"]
    xmin = point_bottomleft.value[0] / fig_w
    ymin = point_bottomleft.value[1] / fig_h

    point_topright = quad["Line1/Point1"]
    xmax = point_topright.value[0] / fig_w
    ymax = point_topright.value[1] / fig_h
    width = xmax - xmin
    height = ymax - ymin

    return (xmin, ymin, width, height)
