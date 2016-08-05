# README

This code was developed during work on Master thesis "Projection method applied to modelling blood flow in cerebral aneurysm". The thesis was writen by Jakub Hrnčíř under the supervision of Jaroslav Hron from Mathematical Institute of Charles University in Prague.

The text of the thesis is available [here](https://drive.google.com/open?id=0B2eGGjPtxvxHTTc0d1JFN0pFMjQ).

## Dependency
The code uses [FEniCS](https://fenicsproject.org/index.html) software. The code was created to work with 1.6.0 release of FEniCS with [PETSc](https://www.mcs.anl.gov/petsc/) linear algebra backend.

## Overview
The motivation of the work, problems solved and methods used in the code are described in the thesis.

## Parallel run
The "real" aneurysm problem can be run in parallel, and was tested to run on up to 960 cores. To estimate reasonable number of used cores, see results of strong scaling tests in the thesis.
The test problem cannot be run in parallel due to the technique of generating analytic solution. The technique could be changed, but we saw no reason, as the test problem runs sufficiently fast on one core.

## Basic usage
The code is operated using command line arguments. Basic structure is:

    python main.py problem_name solver_name mesh total_time time_step

|argument|option/value|explanation|
|:---|:---|:---|
|problem name|"womersley_cylinder"| test problem |
| |"real"| real aneurysm problem |
|solver name| "ipcs1"| for IPCS scheme or its modifications|
| |"direct"|previous solving strategy (Newton method with direct solver)|
|mesh|"cyl_c1", "cyl_c2" or "cyl_c3"|test problem meshes (gradually increasing quality)|
| | "HYK", or "HYK3"| real mesh (coarsest or middle quality)|
|total time|positive float|in seconds, computation will run from 0 to this time|
|time step|positive float|in seconds|

All parameters are used without quotes. Many optional parameters can be appended.

In the thesis tests on third, finest real mesh were conducted. This mesh is not included in this repository because of its size (231 MB). 

## Optional arguments
### General arguments
|argument|option/value|explanation|
|:---|:---|:---|
|-n|any string|name of this run instance|
|-s|'noSave' (default)|do not save any XDMF files during computation|
| |'only_vel'|saves only velocity fields|
| |'doSave'|velocity and its divergence, pressure fields|
| |'diff'|adds difference between computed and analytic solution|
|--onset|float|smoothing of boundary condition time span (default 0.5 s)|
|--saventh|positive integer|save velocity, pressure etc. only in every n-th step during first second|
|--ST|"peak"|saves XDMF files only in the second tenth of each second|
| |"min"|saves only few steps around peak flow|
| | |Both options do not save anything during first second, overrides --saventh.|
|--wss|all|compute wall shear stress in every time step|
| |peak|compute wall shear stress in few steps around peak flow|
|--wss_method|"integral" (default)|computes WSS norm averaged over facets|
| |"expression"|computes WSS vector and norm, does not work for higher number of cores|
|--ffc| |FEniCS form compiler options and optimizations|
| |'uflacs_opt' (default)|takes longer to compile but result in more efficient code|
| |'auto_opt'|alternative to 'uflacs_opt'|
| |'auto'|faster compiling, less efficient code (may be used for debugging)|
|--out|"all"|allows output from all processes|

### For "real" problem
|argument|option/value|explanation|
|:---|:---|:---|
|--nu|positive float|kinematic viscosity (default 3.71 mm<sup>2</sup>/s)|
|-F|positive float|inflow velocity multiplicative factor|
### For "womersley_cylinder" problem
|argument|option/value|explanation|
|:---|:---|:---|
|--nufactor|10.0, 1.0, 0.1 or 0.01|multiplicative factor for kinematic viscosity (default 1.0 value corresponds to 3.71 mm<sup>2</sup>/s)|
|-e|"doEC" (default)/"noEC"|do or do not compute error using analytic solution|
|-F|positive float|inflow velocity multiplicative factor|
|--ic|"zero"|use zero initial conditions|
| |"correct"|use analytic solution as initial conditions|
### For IPCS solver
|argument|option/value|explanation|
|:---|:---|:---|
|--stab|positive float|use stabilisation with chosen multiplicative coefficient|
|--cbc_tau| |use simpler stabilisation parameter|
|--cs| |use "consistent" full SUPG stabilisation (requires --stab)|
|-r| |use rotational pressure correction|
|-B| |do not use BC for corrected velocity|
|-b|"outflow" (default)|use zero pressure boundary condition on outflow|
| |"nullspace"|compute singular (full Neumann) Poisson problem for pressure|
|--fo| |force right outflow rate by prescribing constant pressure gradient on outflows (for -b "nullspace", "real" problem)|
|--laplace| |use Navier-Stokes formulation with laplace(u) instead of 2sym(grad(u))|
|--bcv| |use alternative velocity outflow condition (default "CDN" is sym(grad(u))n = 0)|
| |LAP|grad(u)n = 0|
| |DDN|directional do-nothing condition (for the case of backflow instability) (requires uflacs compiler)|
### For IPCS solver - solver options
|argument|option/value|explanation|
|:---|:---|:---|
|--solP|solver code (default: "cg")|solver used for 2nd step|
|--precP|"hypre_amg", "ilu", "sor" (default)|preconditioner used for 2nd step|
|--precV|preconditioner code, (default: "sor")|preconditioner used for 1st step|
|--precVC|preconditioner code, (default: "sor")|preconditioner used for 3rd step|
|--prv1|integer I, (default: 6)|1st step Krylov solver relative precision = 1E-I|
|--pav1|integer I, (default: 10)|1st step Krylov solver absolute precision = 1E-I|
|--pap'|integer I, (default: 6)|2nd step Krylov solver absolute precision = 1E-I|
|--prp'|integer I, (default: 10)|2nd step Krylov solver relative precision = 1E-I|
|--Vrestart|positive integer|set GMRES restart parameter for 1st step|
|--Prestart|positive integer|set GMRES restart parameter for 2nd step|
|-s|"direct"|use only direct solvers|

## Another options
Following options were not mentioned in the thesis:
Problem "steady_cylinder" is a simpler variant of test problem "womersley_cylinder", with steady parabolic inflow profile instead of pulsating Womersley flow.
Problem "FaC3D_benchmark" was meant to test projection method on 3D benchmark of flow around a cylinder (described [here](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_configuration.html)). It was not finished. It can be run to compute velocity etc., but is missing a code to compute and visualize drag and lift functionals.

# Visualisation of results
Computed results are stored in XDMF files. To open them you can use the [ParaView](http://www.paraview.org/) software.

Scripts for convenient visualisation in ParaView are generated after each successful run. The scripts should work well with ParaView version 4.4. For the scripts to work properly, you should open ParaView from the same directory as the folder with results is located (the scripts use relative paths fo XDMF files). To use them, go to menu Tools/Python Shell and then click 'Run Script'.

# Adding new meshes to be used with "real" problem
You will need mesh in XML format, normals and centerpoints for every outflow and inflow and radii of inscribed circles for the inflows. Then prepare_real_mesh.py can be used to generate .hdf5 and .ini files containing mesh, facet function and information used for generating inflow profiles.

