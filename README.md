sglake-detectability

Author: Aaron Stubblefield (Columbia University, LDEO).

# Introduction
This repository contains FEniCS python code for simulating subglacial lake
filling-draining events. The model is isothermal Stokes flow with nonlinear
("Glen's law") viscosity. Grounding line migration and ice-water/ice-air free
surface evolution are included. The contact conditions that determine whether
ice remains in contact with the bed or goes afloat are enforced with a penalty
functional.

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

# Running the code
To run the code:

1. Start the FEniCS Docker image.

2. In Docker, run the main file from the parent directory: `python3 ./source/main.py`.


# Output

Model output is saved in a *results_tX_HY_CZ* directory, where X is the
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

Model options and parameters can be set in the **params.py** file. For example:


1. *Real-time plotting* is available by setting `realtime_plot = "on"` in the
**params.py** file. This outputs a png called 'surfaces' of the free surface geometry at each
timestep.

2. Parameters of interest in **params.py** that may be modified are `t_period` (period of oscillation)
`nt_per_cycle` (number of timesteps per cycle), `Hght` (ice thickness), and `C`
(sliding law coefficient).

3. The lake volume change timeseries may be modified
in the **hydrology.py** file.
