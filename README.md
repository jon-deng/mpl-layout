
![Project logo](logo.svg)

## Summary

MPLLayout is a package used to create figure layouts for [matplotlib](https://matplotlib.org/) using geometric constraints.

MPLLayout can be used to:

* align figure elements (axes, text label location, x and y axis, etc.),
* specify margins around axes,
* specify the figure size,
* and more!

## Motivation

Matplotlib contains several strategies for creating figure layouts (for example, `GridSpec` and `subplots` for grid-based layouts).
These strategies work well, however, they can be insufficient for complex figures or if you want precise placement of figure elements.
To address this, this package uses geometric primitives and constraints to represent and position figure elements.

It consists of:

* geometric primitives to represent figure elements (for example, a `Quadrilateral` primitive represents a figure, a `Point` represents a text anchor, etc.),
* geometric constraints to position figure elements (for example, `Collinear` constraints align lines, `Length` constraints fix line lengths, etc.),
* a solver to find primitives that satisfy the constraints,
* and utilities to generate matplotlib figures and axes from the primitives.

## Installation

You can install the package from PyPI using

```bash
pip install matplotlib-layout
```

Alternateively, clone the repository into a local drive.
Navigate to the project directory and run

```bash
pip install .
```

The package requires `numpy`, `matplotlib`, and `jax`.

## How to use

The tutorial notebook in `examples/tutorial.ipynb` demonstrates the basic usage of the package and explains some of the commonly used geometric constraints.
Other examples are also given in the `examples` folder.
The notebook at `examples/ten_simple_rules_demo.ipynb` contains an interactive demo to recreate figures from ["Ten Simple Rules For Better Figures"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833) (Rougier, Droettboom and Bourne 2014).

A summary of how to use the package is shown in the figure below.
The process resembles creating and constraining geometry in computer-aided design programs like AutoCAD, SolidWorks, etc.
![Project logo](doc/Summary.svg)

## Contributing

This project is a work in progress so there are likely bugs and missing features.
If you would like to contribute a bug fix, a feature, refactor etc. thank you!
All contributions are welcome.

## Motivation and Similar Projects

A similar project with a geometric constraint solver is [`pygeosolve`](https://github.com/SeanDS/pygeosolve).
There is also another project prototype for a constraint-based layout engine for `matplotlib` [`MplLayouter`](https://github.com/Tillsten/MplLayouter), although it doesn't seem active as of 2023.
