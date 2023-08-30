
## Overview

MPLLayout is a Python package meant to make arrangement of plot elements in Matplotlib easy.

It consists of:

* geometric primitives to represent plot elements (for example, a `Box` primitive can represent a `Bbox` or `Axes` in Matplotlib)
* constraints on primitives used to specify the arrangement of plot elements (for example, constraints on the length of lines, collinearity between lines, locations of points, etc.)
* a constraint solver to solve for the arrangement of primitives that satisfy given constraints
* utilities to generate Matplotlib figures and axes from geometric primitives

## Installation

To install this package, clone the repository into a local drive.
Navigate to the project directory and use
```
pip install .
```

You will also need to have the packages `numpy`, `matplotlib`, and `jax`.

## How to use

Basic usage of the project is shown in the `examples` folder.
In particular, the notebook in 'examples/two_axes.ipynb' illustrates the basic usage of the package.

## Contributing

This project is a work in progress so there are likely bugs and missing features.
If you would like to contribute a bug fix, a feature, refactor etc. thank you!
All contributions are welcome.

## Motivation and Similar Projects

A similar project with a geometric constraint solver is [`pygeosolve`](https://github.com/SeanDS/pygeosolve).
