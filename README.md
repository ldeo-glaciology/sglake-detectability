sglake-detectability

# Overview
This repository contains FEniCS python code for simulating subglacial lake
filling-draining events. The model is 2D isothermal Stokes flow with nonlinear
("Glen's law") viscosity. Grounding line migration and ice-water/ice-air free
surface evolution are included. The contact conditions that determine whether
ice remains in contact with the bed or goes afloat are enforced with a penalty
functional. I am using this model to study the "detectability" of subglacial
lake filling/draining events from surface observations.

There is also
a linearized (small-perturbation) version of the model that is used to
produce water volume change and lake length estimates (see *linear-model* description below).

# Dependencies
## Required dependencies
This code runs with the current FEniCS Docker image (https://fenicsproject.org/download/).
Docker may be obtained at: https://www.docker.com/. To run the Docker image:

`docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current`

## Optional dependencies

1. FFmpeg (https://www.ffmpeg.org/) can be used, along with **make_movie.py**,
to create a video of the evolving free surface geometry over time. See description below.

2. ParaView is useful for visualizing velocity/pressure solutions to the Stokes equations (https://www.paraview.org/).

# Contents

## 1. Source files
The model is organized in 7 python files in the *source* directory as follows.

1. **geometry.py** contains the geometric description of the bed and initial ice-water interface.

2. **params.py** contains all of the model parameters and model options.

3. **stokes.py** contains the Stokes system solver and related functions.

4. **meshfcns.py** contains functions that solve the surface kinematic equations, move the mesh,
    and return the grounding line positions.

5. **boundaryconds.py** contains functions that mark the mesh boundary and apply boundary conditions.

6. **hydrology.py** contains the subglacial lake volume change timeseries.

7. **main.py** runs the model. It contains the time-stepping loop that
calls the Stokes solver and mesh-related functions at each timestep.

## 2. Scripts

The *scripts* directory contains:

1. **plot.py**: (1) Creates plots of the free surface geometry at lake lowstand
and highstand and (2) computes the estimated volume change from surface displacement
and compares this to the true lake volume change.

2. **make_movie.py**: generates .png
images of the basal and upper surfaces for each time step. These may then be
used to create a .mp4 movie using the FFmpeg command in
the comments at the top of the **make_movie.py** file:
`ffmpeg -r 50 -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4`

These scripts are run from the parent directory.

## 3. Linear model
The *linear-model* directory contains two files that are based on a small-perturbation
approximation of the free surface model. The model is very computationally efficient
in this context, relying only on Fourier transforms and quadrature. They work
with the lastest version of SciPy (1.6).
The files are:

1. **ratios.py**: Constructs volume change and lake length estimates
relative to their true values for a given range of ice thickness, oscillation period,
and basal friction coefficient.

2. **map.py**: Computes the minimum detectable lake length as a function of
the ice thickness and basal friction coefficient. This can then be used to convert
maps of the ice thickness and basal friction into a map of the minimum detectable
lake length upon setting some other parameters (primarily oscillation amplitude and period).

These files are run by: `python3 filename.py`.

The data used to construct the map is from the publication:
>Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within the Antarctic ice sheet and its controls inferred from satellite observations. Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.

Plots of the results are automatically produced and saved as png's.

A Jupyter notebook for the running **map.py** is implemented in **PlotMap.ipynb**.

This can be run with Pangeo-Binder via the link:
[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/pangeo-gallery/default-binder/master/?urlpath=git-pull?repo=https://github.com/ldeo-glaciology/sglake-detectability.git%26amp%3Burlpath=lab/tree/sglake-detectability.git/linear-model/PlotMap.ipynb)


Note: this model assumes a constant value for the ice viscosity. The default
value (10^12 Pa s) produces results that agree quite well with the nonlinear free-surface model.

# Running the FEniCS code
To run the FEniCS code:

1. Start the FEniCS Docker image.

2. In Docker, run the main file from the parent directory: `python3 ./source/main.py`

This runs the code with the default options.
Read below for a discussion of model options.

# Output

Model output from the FEniCS code is saved in a *results_tX_HY_CZ* directory, where X is the
oscillation period in years, Y is the ice thickness in km, and Z = log_10 (C),
where C is the sliding law coefficient (C=10^Z, units of Pa s/m). The directory includes

1. the Stokes solution (*stokes* subdirectory),

2. upper and lower surface geometry at each timestep (Gamma_s and Gamma_h, resp.),

3. minimum and maximum grounding line positions (x_left and x_right, resp.),

4. penalty functional residual (P_res),

5. mean elevation of upper and lower surfaces over the ice-water interface (s_mean and h_mean, resp.),

6. spatial coordinate (X),

7. time coordinate (t),

8. lake volume timeseries (lake_vol), and

9. deviation of mean water pressure from cryostatic pressure (dPw, in kilopascals).

Gamma_s and Gamma_h are two-dimensional arrays:
the columns are the free surfaces at the timestep corresponding to the row index.
The spatial coordinate X is finer than the mesh spacing because Gamma_s and Gamma_h
are created by linear interpolation of the mesh nodes in SciPy.

# Model options

FEniCS model options and parameters are set on the command line. To see a list of model options,
run `python3 ./source/main.py -h`.  Most importantly:

1. Parameters of interest are `pd` (period of oscillation, yr),
`H` (ice thickness, meters), `C`
(basal drag coefficient, Pa s/m), and `L` (domain length, km).

2. *Real-time plotting* is available by setting `-plotting on`.
This outputs a png called 'surfaces' of the free surface geometry at each
timestep.

Other parameters can be modified in **params.py**.
