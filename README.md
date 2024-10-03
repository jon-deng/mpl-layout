
![Project logo](logo.svg)

## Summary

MPLLayout is a package used to create figure layouts in `matplotlib` with geometric constraints.

You can use MPLLayout to:

* consistently size margins around axes
* align visual elements within a figure, such as the edges of axes, label positions, etc.
* fix the size of axes of figures
* and more!

## Motivation

`matplotlib` contains a few strategies for creating figure layouts, for example, `GridSpec` for grid-based layouts of axes or `plt.subplots`.
These layout strategies are very useful but follow a predetermined strategy for arranging axes elements which is not easily customizable by users.
This can make it difficult to achieve precise placement of axes, or sometimes leave extraneous whitespace.
To achieve more precise placement MPLLayout provides a set of geometric constraints and primitives to model the positioning of axes, figures, and other `matplotlib` visual elements.
It consists of:

* geometric primitives to represent plot elements (for example: a `Quadrilateral` primitive represents  `Figure` or `Axes` frames, a `Point` can represent a text anchor, etc.)
* geometric constraints to specify the arrangement of plot elements (for example: constraints on the length of lines, collinearity between lines, locations of points, dimensions of rectangles, etc.)
* a constraint solver to solve for the arrangement of primitives that satisfy given constraints
* utilities to generate `matplotlib` figures and axes from geometric primitives

## Installation

To install this package, clone the repository into a local drive.
Navigate to the project directory and use
```
pip install .
```

You will also need to have the packages `numpy`, `matplotlib`, and `jax`.

## How to use

Basic usage of the project is shown in the `examples` folder.
The notebook in 'examples/ten_simple_rules_demo.ipynb' gives an interactive demo.

## Contributing

This project is a work in progress so there are likely bugs and missing features.
If you would like to contribute a bug fix, a feature, refactor etc. thanks!
All contributions are welcome.

## Motivation and Similar Projects

A similar project with a geometric constraint solver is [`pygeosolve`](https://github.com/SeanDS/pygeosolve).
There is also another project prototype for a constraint-based layout engine for `matplotlib` [`MplLayouter`](https://github.com/Tillsten/MplLayouter), although it doesn't seem active as of 2023.
