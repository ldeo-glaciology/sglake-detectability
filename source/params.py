# All model parameters and options are recorded here.
import numpy as np
#-------------------------------------------------------------------------------
#-----------------------------MODEL OPTIONS-------------------------------------

# Turn 'on' or 'off' real-time plotting that saves a png figure called 'surfs' at
# each time step of the free surface geometry.
realtime_plot = 'on'

# Turn 'on' or 'off' Newton convergence information:
print_convergence = 'off'

#-------------------------------------------------------------------------------
#-----------------------------MODEL PARAMETERS----------------------------------

# Material parameters
A0 = 3.1689e-24                    # Glen's law coefficient (ice softness)
n = 3.0                            # Glen's law exponent
rm2 = 1 + 1.0/n - 2.0              # exponent in variational forms: r-2
B0 = A0**(-1/n)                    # ice hardness
B = (2**((n-1.0)/(2*n)))*B0        # coefficient in weak form
rho_i = 917.0                      # density of ice
rho_w = 1000.0                     # density of water
g = 9.81                           # gravitational acceleration
C = 1.0e8                          # sliding law friction coefficient
Cexp = str(int(np.floor(np.log10(C))))

# Numerical parameters
eps_v = 1.0e-15                    # flow law regularization parameter
eps_p = 1.0e-13                    # penalty method parameter
quad_degree = 16                   # quadrature degree for weak forms

tol = 1.0e-2                       # numerical tolerance for boundary geometry:
                                   # s(x,t) - b(x) > tol on ice-water boundary,
                                   # s(x,t) - b(x) <= tol on ice-bed boundary.

# Geometry/mesh parameters
Hght = 1000.0                      # (initial) height of the domain
Lngth = 100*1000.0                 # length of the domain
Ny = int(Hght/500.0)               # number of elements in vertical direction
Nx = int(Lngth/500.0)              # number of elements in horizontal direction

# Time-stepping parameters
t_final = 5.0*3.154e7             # final time (yr*sec_per_year).
t_period = t_final/2.0             # oscillation period
nt_per_cycle = 250                 # number of timesteps per oscillation
nt = int(t_final/t_period*nt_per_cycle) # number of time steps
dt = t_final/nt                    # timestep size

nx = 10000                         # number of grid points for interpolating
                                   # free surfaces and plotting (larger
                                   # than true number elements Nx)

X_fine = np.linspace(0,Lngth,nx)   # horizontal coordinate for computing surface
                                   # slopes and plotting.

# Choose boundary conditions on inflow/outflow boundaries
#wall_bcs = 'neumann'              # cryostatic stress
wall_bcs = 'dirichlet'             # zero inflow/outflow
