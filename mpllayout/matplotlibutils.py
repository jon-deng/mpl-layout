"""
Create a single axes figure
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

    width, height = wh_from_box(prims['Figure'])

    print(rect_from_box(prims['Axes1'], fig_size=(width, height)))

    fig = plt.Figure((width, height))
    axs = {
        key: fig.add_axes(rect_from_box(prim, (width, height)))
        for key, prim in prims.items()
        if 'Axes' in key and key.count('.') == 0
    }
    return fig, axs

def wh_from_box(box: geo.Box):
    """
    Return the width and height of a `Box`
    """

    point_bottomleft = box.prims[0]
    xmin = point_bottomleft.param[0]
    ymin = point_bottomleft.param[1]

    point_topright = box.prims[2]
    xmax = point_topright.param[0]
    ymax = point_topright.param[1]

    return (xmax-xmin), (ymax-ymin)

def rect_from_box(box: geo.Box, fig_size=(1, 1)):
    """
    Return the lower left corner and width and height of a `Box`
    """
    fig_w, fig_h = fig_size

    point_bottomleft = box.prims[0]
    xmin = point_bottomleft.param[0]/fig_w
    ymin = point_bottomleft.param[1]/fig_h

    point_topright = box.prims[2]
    xmax = point_topright.param[0]/fig_w
    ymax = point_topright.param[1]/fig_h

    return (xmin, ymin, (xmax-xmin), (ymax-ymin))
