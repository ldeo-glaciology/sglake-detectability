# All model parameters and options are set here.

import sys,argparse

parser = argparse.ArgumentParser(\
    description='Set parameters for subglacial lake problem.')
parser.add_argument('-C', type=float, default=1.0e9, metavar='drag_coeff',
                    help='Basal drag coefficient (Pa s/m)')
parser.add_argument('-H', type=float, default=1000.0, metavar='thickness',
                    help='Ice thickness (m)')
parser.add_argument('-pd', type=float, default=2, metavar='period',
                    help='Filling/draining oscillation period (yr)')
parser.add_argument('-plotting', type=str, default='off', metavar='on/off',
                    help='Turn real-time plotting \'on\' or \'off\' ')
parser.add_argument('-print_info', type=str, default='off', metavar='on/off',
                    help='Turn \'on\' to print Newton convergence info ')
parser.add_argument('-save_vtk', type=str, default='off', metavar='on/off',
                    help='Turn \'on\' to save Stokes vtk files ')
args, unknown = parser.parse_known_args()

import numpy as np
#-------------------------------------------------------------------------------
#-----------------------------MODEL OPTIONS-------------------------------------

# Turn 'on' or 'off' real-time plotting that saves a png figure called 'surfs' at
# each time step of the free surface geometry.
realtime_plot = args.plotting

# Turn 'on' or 'off' Newton convergence information:
print_convergence = args.print_info

# save vtk files for stokes solution if 'on':
save_vtk = args.save_vtk

#-------------------------------------------------------------------------------
#-----------------------------MODEL PARAMETERS----------------------------------
# physical units:
# time - seconds
# space - meters
# pressure - pascals
# mass - kg

# material parameters
A0 = 3.1689e-24                    # Glen's law coefficient (ice softness, Pa^{-n}/s)
n = 3.0                            # Glen's law exponent
rm2 = 1 + 1.0/n - 2.0              # exponent in variational forms: r-2
B0 = A0**(-1/n)                    # ice hardness (Pa s^{1/n})
B = (2**((n-1.0)/(2*n)))*B0        # coefficient in weak form (Pa s^{1/n})
rho_i = 917.0                      # density of ice (kg/m^3)
rho_w = 1000.0                     # density of water (kg/m^3)
g = 9.81                           # gravitational acceleration (m/s^2)
C = args.C                         # sliding law friction coefficient (Pa s/m)

# numerical parameters
eps_p = 1.0e-13                    # penalty method parameter for unilateral condition
eps_v = 1.0e-15                    # flow law regularization parameter

quad_degree = 20                   # quadrature degree for weak forms

tol = 1.0e-2                       # numerical tolerance for boundary geometry:
                                   # s(x,t) - b(x) > tol on ice-water boundary,
                                   # s(x,t) - b(x) <= tol on ice-bed boundary.

# geometry/mesh parameters
Hght = args.H                      # (initial) height of the domain (m)
Lngth = 100*1000.0                 # length of the domain (m)

Ny = int(Hght/500.0)               # number of elements in vertical direction
Nx = int(Lngth/500.0)              # number of elements in horizontal direction


# time-stepping parameters
t_period = args.pd*3.154e7         # oscillation period (secs; yr*sec_per_year)
t_final = 2.0*t_period             # final time
nt_per_cycle = 500                 # number of timesteps per oscillation
nt = int(t_final/t_period*nt_per_cycle) # number of time steps
dt = t_final/nt                    # timestep size

# spatial coordinate for plotting and interpolation

nx = 10000                         # number of grid points for interpolating
                                   # free surfaces and plotting (larger
                                   # than true number elements Nx)

X_fine = np.linspace(0,Lngth,nx)   # horizontal coordinate for computing surface
                                   # slopes and plotting.
