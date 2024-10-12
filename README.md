
![Project logo](logo.svg)

## Summary

MPLLayout is a package used to create figure layouts for [matplotlib](https://matplotlib.org/) using geometric constraints.

MPLLayout can be used to:

* align figure components (such as axes sides or text label locations),
* specify margins around axes,
* specify relative sizes of axes,
* and more!

## Motivation

Matplotlib contains a few strategies for creating figure layouts, for example, `GridSpec` and `subplots` for grid-based axes layouts.
These strategies are very useful but follow a predetermined strategy for arranging axes that is not easily customizable.
This sometimes makes it difficult to precisely place axes or can result in extra whitespace.
To address this, MPLLayout uses geometric constraints and primitives to precisely position figures, axes and other elements in matplotlib.

It consists of:

* geometric primitives to represent plot elements (for example, a `Quadrilateral` primitive can represent a figure or axes, a `Point` primitive can represent a text anchor),
* geometric constraints to specify arrangements of plot elements (for example, axes sides can be aligned with `Collinear` constraints, line lengths can be fixed with `Length` constraints),
* a solver to find geometric primitives that satisfy the constraints,
* and utilities to generate `matplotlib` figures and axes from geometric primitives.

## Installation

To install this package, clone the repository into a local drive.
Navigate to the project directory and run
```
pip install .
```

The package also requires `numpy`, `matplotlib`, and `jax`.

## How to use

Basic usage of the package is shown in the `examples` folder.
The notebook at `examples/ten_simple_rules_demo.ipynb` contains an interactive demo using MPLLayout to recreate figures from ["Ten Simple Rules For Better Figures"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833) (Rougier, Droettboom and Bourne 2014).
This demo splits the layout creation across several steps to better explain the result of different geometric constraints.

## Contributing

This project is a work in progress so there are likely bugs and missing features.
If you would like to contribute a bug fix, a feature, refactor etc. thank you!
All contributions are welcome.

## Motivation and Similar Projects

A similar project with a geometric constraint solver is [`pygeosolve`](https://github.com/SeanDS/pygeosolve).
There is also another project prototype for a constraint-based layout engine for `matplotlib` [`MplLayouter`](https://github.com/Tillsten/MplLayouter), although it doesn't seem active as of 2023.
