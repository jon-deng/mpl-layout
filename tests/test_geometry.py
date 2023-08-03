"""
Test geometric primitive and constraints 
"""

import pytest
from pprint import pprint

import numpy as np

from mpllayout import solver, geometry as geo

class TestConstraints:

    def test_point_collection(self):
        layout = solver.Layout()

        layout.add_prim(geo.Point([0, 0]), 'Origin')
        layout.add_prim(geo.Point([1, 1]))
        layout.add_prim(geo.Point([2, 2]))

        print(layout.prims.keys())

        layout.add_constraint(
            geo.PointLocation(np.array([0, 0])), 
            ('Origin',)
        )

        layout.add_constraint(
            geo.PointToPointAbsDistance(5.0, np.array([1, 0])), 
            ('Origin', 'Point1')
        )
        layout.add_constraint(
            geo.PointToPointAbsDistance(4.0, np.array([0, 1])), 
            ('Origin', 'Point1')
        )

        layout.add_constraint(
            geo.PointToPointAbsDistance(-1.0, np.array([1, 0])), 
            ('Point1', 'Point2')
        )
        layout.add_constraint(
            geo.PointToPointAbsDistance(2.0, np.array([0, 1])), 
            ('Point1', 'Point2')
        )

        new_prims, info = solver.solve(
            layout.prims, layout.constraints, layout.constraint_graph
        )
        pprint(new_prims)
        pprint(info)

    def test_box(self):

        layout = solver.Layout()

        # layout.add_prim(geo.Point([0, 0]), 'Origin')

        xmin, xmax = 1, 4
        ymin, ymax = 1, 2
        box_points = [
            geo.Point([xmin+0.1, ymin+0.5]),
            geo.Point([xmax+0.6, ymin]),
            geo.Point([xmax, ymax]),
            geo.Point([xmin-0.1, ymax])
        ]
        box_lines = [
            geo.LineSegment(prims=(pointa, pointb))
            for pointa, pointb in zip(box_points, box_points[1:]+box_points[:1])
        ]
        layout.add_prim(geo.Box(prims=box_lines))
        print(layout.constraints)
        print(layout.constraint_graph)
        # assert False

        layout.add_constraint(
            geo.PointLocation(np.array([0, 0])), 
            ('Box0.LineSegment0.Point0',)
        )

        # layout.add_constraint(
        #     geo.PointToPointAbsDistance(1.2, np.array([1, 0])), 
        #     ('Origin', 'Box0.LineSegment0.Point0')
        # )
        # layout.add_constraint(
        #     geo.PointToPointAbsDistance(1.2, np.array([0, 1])), 
        #     ('Origin', 'Box0.LineSegment0.Point0')
        # )

        layout.add_constraint(
            geo.PointToPointAbsDistance(5.2, np.array([1, 0])), 
            ('Box0.LineSegment0.Point0', 'Box0.LineSegment1.Point1')
        )
        layout.add_constraint(
            geo.PointToPointAbsDistance(2.4, np.array([0, 1])), 
            ('Box0.LineSegment0.Point0', 'Box0.LineSegment1.Point1')
        )

        # layout.add_constraint(
        #     geo.Horizontal(), 
        #     ('Box0.LineSegment0',)
        # )

        # layout.add_constraint(
        #     geo.Vertical(), 
        #     ('Box0.LineSegment1',)
        # )

        pprint("Constraints")
        pprint(layout.constraints.keys())

        new_prims, info = solver.solve(
            layout.prims, layout.constraints, layout.constraint_graph
        )
        pprint(new_prims)
        pprint(info) 
