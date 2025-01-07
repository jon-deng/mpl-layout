"""
Utilities for creating `matplotlib` elements from geometric primitives
"""

from typing import Optional
from numpy.typing import NDArray
import warnings

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import primitives as pr
from . import constraints as cr

# NOTE: Use special primitive classes rather than keys to determine figure/axes?
# If you do, this should be done for both `subplots` and `update_subplots`
def subplots(
    root_prim: pr.Primitive,
    fig_key: str = "Figure",
    axs_keys: Optional[list[str]] = None,
) -> tuple[Figure, dict[str, Axes]]:
    """
    Create matplotlib `Figure` and `Axes` objects from geometric primitives

    The `Figure` and `Axes` objects are extracted based on labels in the primitive tree
    and have sizes and positions from their corresponding primitives.

    Parameters
    ----------
    root_prim: pr.Primitive
        The root primitive
    fig_key: str
        The quadrilateral key corresponding to the figure

        The key is "Figure" by default.
    axs_keys: Optional[list[str]]
        Axes keys

        If supplied, only these axes keys will be used to generate `Axes` instances.

    Returns
    -------
    fig: Figure
        The matplotlib `Figure`
    axs: dict[str, Axes]
        The matplotlib `Axes` instances
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
    root_prim: pr.Primitive, fig_key: str, fig: Figure, axs: dict[str, Axes],
):
    """
    Update matplotlib `Figure` and `Axes` object positions from primitives

    The `Figure` and `Axes` objects are extracted based on labels in the primitive tree
    and have sizes and positions updated from their corresponding primitives.

    Parameters
    ----------
    root_prim: pr.Primitive
        The root primitive
    fig_key: str
        The quadrilateral key in `root_prim` corresponding to the figure
    fig: Figure
        The `Figure` to update
    axs: dict[str, Axes]
        The `Axes` objects to update

    Returns
    -------
    fig: Figure
        The updated matplotlib `Figure`
    axs: dict[str, Axes]
        The updated matplotlib `Axes` instances
    """
    # Set Figure position
    quad = root_prim[fig_key]
    fig_origin = quad['Line0/Point0'].value
    fig_size = np.array(width_and_height_from_quad(quad))
    fig.set_size_inches(fig_size)

    # Set Axes properties/position
    for key, ax in axs.items():
        # Set Axes dimensions
        quad = root_prim[f"{key}/Frame"]
        ax.set_position(rect_from_box(quad, fig_origin, fig_size))

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

    return fig, axs

def find_axis_position(axes_frame: pr.Quadrilateral, axis: pr.Quadrilateral) -> str:
    """
    Return the axis position relative to a frame

    Parameters
    ----------
    axes_frame: pr.Quadrilateral
        The axes frame
    axis: pr.Quadrilateral
        The axes axis

        This can be any x or y axis

    Returns
    -------
    position: str
        One of ('bottom', 'top', 'left', 'right') indicating the axis position
    """
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
    Return the width and height of a quadrilateral

    Parameters
    ----------
    quad: pr.Quadrilateral

    Returns
    -------
    tuple[float, float]
        The width and height
    """

    coord_botleft = quad["Line0/Point0"].value
    xmin, ymin = coord_botleft

    coord_topright = quad["Line1/Point1"].value
    xmax, ymax = coord_topright

    return (xmax - xmin), (ymax - ymin)


def rect_from_box(
    quad: pr.Quadrilateral,
    fig_origin: NDArray,
    fig_size: NDArray = np.array((1, 1))
) -> tuple[float, float, float, float]:
    """
    Return a `rect' tuple, `(left, bottom, width, height)`, from a quadrilateral

    This tuple of quadrilateral information can be used to create a `Bbox` or `Axes`
    object in `matplotlib`.

    Parameters
    ----------
    quad: pr.Quadrilateral
        The quadrilateral
    fig_origin: NDArray
        Coordinates for the figure bottom left corner
    fig_size: NDArray
        The width and height of the figure

        This should be supplied so that the rect tuple has units relative to the figure.
        Some matplotlib `Axes` constructors accept the rect tuple in figure units by default.

    Returns
    -------
    xmin, ymin, width, heigth: tuple[float, float, float, float]
    """

    coord_botleft = quad["Line0/Point0"].value
    xmin, ymin = (coord_botleft-fig_origin) / fig_size

    coord_topright = quad["Line1/Point1"].value
    xmax, ymax = (coord_topright-fig_origin) / fig_size
    width = xmax - xmin
    height = ymax - ymin

    return (xmin, ymin, width, height)
