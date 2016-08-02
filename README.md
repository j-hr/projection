# README

This code was developed during work on Master thesis "Projection method applied to modelling blood flow in cerebral aneurysm". The thesis was writen by Jakub Hrnčíř under the supervision of Jaroslav Hron from Mathematical Institute of Charles University in Prague.

The text of the thesis is available [here](https://drive.google.com/open?id=0B2eGGjPtxvxHTTc0d1JFN0pFMjQ).

## Dependency
The code uses [FEniCS](https://fenicsproject.org/index.html) software. The code was created to work with 1.6.0 and 2016.1 release of FEniCS with [PETSc](https://www.mcs.anl.gov/petsc/) linear algebra backend.

## Overview
The motivation of the work, problems solved and methods used in the code are described in the thesis.

## Parallel run
The "real" aneurysm problem can be run in parallel, and was tested to run on up to 960 cores. To estimate reasonable number of used cores, see results of strong scaling tests in the thesis.
The test problem cannot be run in parallel due to the technique of generating analytic solution. The technique could be changed, but we saw no reason, as the test problem runs sufficiently fast on one core.

## Basic usage
The code is operated using command line arguments. Basic structure is:
    python main.py problem_name solver_name mesh total_time time_step

|---|
|problem name|"womersley_cylinder" for test problem or "real" for real aneurysm problem   |
|solver name|  "ipcs1" for IPCS scheme or its modifications or "direct" for previous solving strategy |
|mesh|"cyl_c1", "cyl_c2" or "cyl_c3" for test problem meshes (gradually increasing quality)|
| | "HYK", or "HYK3" for real mesh (coarsest or middle quality)|
|total time|in seconds, computation will run from 0 to this time (e. g. "1", or "0.25")|
|time step|in seconds, e. g. "0.1"|

All parameters are used without quotes. Many optional parameters can be appended.

In the thesis tests on third, finest real mesh were conducted. This mesh is not included in this repository because of its size (231 MB). 

## Optional arguments
### General arguments
|argument|options|explanation|
|---|
|-n|any string|name of this run instance|
|-s|'noSave' (default), 'doSave', 'diff', 'only_vel'|what values are stored in XDMF files during computation: "only_vel" saves only velocity fields, "doSave" adds pressure and divergence of velocity, "diff" adds difference between computed and analytic solution if available|
|--saventh|positive integer|save velocity, pressure etc. only in every n-th step during first second|
|--ST|"peak","min"|to save disk space, "peak" saves XDMF files only in second tenth of each second, "min" saves only up to ten steps around peak. Both options do not save anything during first second. Overrides --saventh.|
|--ffc|'auto_opt', 'uflacs', 'uflacs_opt' (default), 'auto'|FEniCS form compiler options and optimizations (optimized takes longer to compile but result in more efficient code)|
|--out|"all", "main" (default)|allows output from all processes or only the main one|
### For "real" problem

### For "womersley_cylinder" problem

### For IPCS solver

### For direct solver

## Another options
Following options were not mentioned in the thesis.
Problem "steady_cylinder" is a simpler variant of test problem "womersley_cylinder", with steady parabolic inflow profile instead of pulsating Womersley flow.
Problem "FaC3D_benchmark"

# Visualisation of results
Computed results are stored in XDMF files. To open them you can use the [ParaView](http://www.paraview.org/) software.

Scripts for convenient visualisation in ParaView are generated after each successful run. The scripts should work well with ParaView version 4.4. For the scripts to work properly, you should open ParaView from the same directory as the folder with results is located (the scripts use relative paths fo XDMF files). To use them, go to menu Tools/Python Shell and then click 'Run Script'.



