# All model parameters and options are recorded here.
import numpy as np
#-------------------------------------------------------------------------------
#-----------------------------MODEL OPTIONS-------------------------------------

# Turn 'on' or 'off' real-time plotting that saves a png figure called 'surfs' at
# each time step of the free surface geometry.
realtime_plot = 'off'

# Turn 'on' or 'off' Newton convergence information:
print_convergence = 'off'

# save vtk files for stokes solution if 'on':
save_vtk = 'off'

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
C = 1.0e10                         # sliding law friction coefficient (Pa s/m)

# numerical parameters
eps_v = 1.0e-15                    # flow law regularization parameter
eps_u = 1.0e-13                    # penalty method parameter for unilateral condition
eps_b = 1.0e-20                    # penalty method parameter for bilateral condition

quad_degree = 20                   # quadrature degree for weak forms

tol = 1.0e-2                       # numerical tolerance for boundary geometry:
                                   # s(x,t) - b(x) > tol on ice-water boundary,
                                   # s(x,t) - b(x) <= tol on ice-bed boundary.

# geometry/mesh parameters
Hght = 1000.0                      # (initial) height of the domain (m)
Lngth = 100*1000.0                 # length of the domain (m)

basin_width = 15*1000.0            # half-width of lake basin
                                   # (sets how far grounding lines can migrate)

Ny = int(Hght/500.0)               # number of elements in vertical direction
Nx = int(Lngth/500.0)              # number of elements in horizontal direction


# time-stepping parameters
t_period = 2.5*3.154e7             # oscillation period (secs; yr*sec_per_year)
t_final = 2.0*t_period             # final time
nt_per_cycle = 250                 # number of timesteps per oscillation
nt = int(t_final/t_period*nt_per_cycle) # number of time steps
dt = t_final/nt                    # timestep size

# spatial coordinate for plotting and interpolation

nx = 10000                         # number of grid points for interpolating
                                   # free surfaces and plotting (larger
                                   # than true number elements Nx)

X_fine = np.linspace(0,Lngth,nx)   # horizontal coordinate for computing surface
                                   # slopes and plotting.

# choose side-wall boundary condition type
wall_bcs = "cryostatic"     # no net horizontal inflow prescribed:
                            # cryostatic normal stress and zero vertical velocity
                            # on inflow/outflow boundaries

#wall_bcs = "throughflow"   # constant horizontal inflow speed (plug flow)
                            # zero vertical shear stress on inflow & outflow
                            # cryostatic normal stress on outflow boundary

U0 = 500.0/3.154e7          # inflow speed for throughflow bc's
                            # note: this should be "compatible" with the
                            # basal drag coeffcient C. that is, choosing
                            # a huge C and huge U0 is inadvisable.
