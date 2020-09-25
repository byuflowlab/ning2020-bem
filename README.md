# Reproducing Blade Element Momentum Paper

This repository contains scripts to reproduce the results shown in ["Using Blade Element Momentum Methods with Gradient-Based Design Optimization"](http://flow.byu.edu/publications/)

Note that some of the timings/plots are slightly different from that in the paper.  Between submitting the paper and creating this repo I've updated Julia and several packages.  However, the conclusions remain the same.

- roots.jl: demonstrations of failure with 2D methodology and convergence with 1D
- derivatives.jl: time to construct Jacobian for AD and sparse AD (and sparse FD)
- optimize.jl: setup and solve optimize problem.  timings for different approaches.
- hover.jl: rotorcraft validation

A Manifest.toml file is added to the repo to aid in reproducibility.  By [activating and instantiating the project](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1) you will have the exact same versions of packages that I used.  The only exception may be Snopt (optimizer used in optimize.jl) as that is a commercial tool and requires a separate license.


