# map.py: simple method for estimating observed vs. true water volume change,
# lake size, and timing of subglacial lake filling-draining cycles
#
# OVERVIEW
# this code constructs a maps of the estimated vs. true water volume change,
# lake size, and highstand/lowstand times of subglacial lake filling-draining
# cyces, given maps of the ice thickness and basal friction coefficient.
#
# the computation is based on a small-perturbation ice-flow model
# --see the supporting information for a full description of the method.
#
# the main parameters that can be set below are:
#  (1) the subglacial lake length (L_true)
#  (2) the subglacial lake oscillation period (t_pd)
#  (3) the maximum amplitude of the oscillation (amp)
#  (4) ice viscosity (eta)
#
# DATA
# the ice thickness and basal friction maps used here (by default) were
# obtained by personal communication between J. Kingslake (Columbia University)
# and R. Arthern (British Antarctic Survey), and featured in the publication:
#
#   Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within
#   the Antarctic ice sheet and its controL_true inferred from satellite observations.
#   Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,interpolate
from scipy import integrate
from scipy.fft import fft,ifft,fftshift,fftfreq
import matplotlib as mpl
import numpy as np
import copy
import xarray as xr
import gcsfs

# 1.--------------------FUNCTIONS FOR PROCESSING THE DATA-----------------------
def get_Dj(lamda,beta_nd,w_ft,k):
    # function for computing displacements D1 (in-phase with base) and
    # D2 (out-of-phase with base)
    g = beta_nd/k

    # relaxation function
    R1 =  (1/k)*((1+g)*np.exp(4*k) - (2+4*g*k)*np.exp(2*k) + 1 -g)
    D = (1+g)*np.exp(4*k) + (2*g+4*k+4*g*(k**2))*np.exp(2*k) -1 + g
    R = R1/D

    # transfer function
    T1 = 2*(1+g)*(k+1)*np.exp(3*k) + 2*(1-g)*(k-1)*np.exp(k)
    T = T1/D

    G1 = T*w_ft
    G2 = 1 + (lamda*R)**2

    # displacements
    D1 = ifft(G1/G2).real
    D2 = ifft(lamda*R*G1/G2).real

    return D1,D2

def get_Tj(D1,D2,x):
    # times where elevation anomaly is maximized (T1) and minimized (T2)
    # these are the estimated highstand/lowstand times
    T1 = np.pi - np.arctan(np.max(D2)/np.max(D1))
    T2 = 2*np.pi - np.arctan(np.max(D2)/np.max(D1))

    return T1,T2

def get_kappaj(T1,T2):
    # weights on the displacement functions:
    # kappa1 is the in-phase component
    # kappa2 is the out-of-phase component
    kappa1 = np.cos(T2) - np.cos(T1)
    kappa2 = np.sin(T1) - np.sin(T2)

    return kappa1,kappa2

def get_ratios(H,t_pd,beta_d,L_true):
    # compute ratios of the estimated lake length (dL) and water volume change (dV)
    # relative to their true values, as well as the oscillation phase lag (lag)
    # given the true lake length (L_true), dimensional friction (beta_d), and
    # ice thickness (H)

    # discretization in frequency domain
    N = 2000
    x = np.linspace(-100,100,num=N)     # domain is 200 ice thicknesses long
    d = np.abs(x[1]-x[0])
    k = fftfreq(N,d)   # frequency
    k[0] = 1e-10       # set zero frequency to small number due to (integrable) singularity
    k *= 2*np.pi       # convert to SciPy's Fourier transform definition (from angular
                       # frequency version used in manuscript)


    # function for spatial component of basal vertical velocity anomaly (w)
    # (default is a Gaussian)
    sigma = (L_true/H)/3                   # define standard deviation for Gaussian
    w = np.exp(-0.5*(x/sigma)**2)

    w_ft = fft(w)                          # fourier transform (for numerical method)

    beta_nd = beta_d*H/(2*eta)             # non-dimensional friction parameter
                                           # relative to viscosity/ice thickness

    tr =  (4*np.pi*eta)/(rho*g*H)          # relaxation time
    lamda = t_pd/tr                        # ratio of oscillation period to relaxation time

    D1,D2 = get_Dj(lamda,beta_nd,w_ft,k)   # compute surface displacements

    T1,T2 = get_Tj(D1,D2,x)                # compute estimated highstand/lowstand times

    kappa1,kappa2 = get_kappaj(T1,T2)      # compute weights for displacements

    dH = kappa1*D1 + kappa2*D2             # compute surface elevation change anomaly

    dS = 2*w                               # elevation change at base


    # interpolate displacements for integration
    dSi = interpolate.interp1d(x, dS,fill_value="extrapolate")
    dHi = interpolate.interp1d(x, dH,fill_value="extrapolate")

    # compute true volume change
    dV_true = integrate.quad(dSi,-0.5*L_true/H,0.5*L_true/H,full_output=1)[0]

    # compute estimated lake length
    if np.size(x[np.abs(dH)>delta])>0:
        # if displacement exceeds noise threshold, lake length can be estimated
        x0 = x[np.abs(dH)>delta]
    else:
        # otherwise, lake is not detectable
        x0 = 0*x

    # compute estimated lake length
    L_est = 2*np.max(x0)                 # problem is symmetric with respect to x

    if L_est > 1e-5:
        # compute estimated volume change
        dV_est = integrate.quad(dHi,-0.5*L_est,0.5*L_est,full_output=1)[0]

        dV = dV_est/dV_true             # volume change ratio
        dL = L_est*H/L_true             # lake length ratio
        lag = (2/np.pi)*(np.pi-T1)      # phase lag

    else:
        dV = 0
        dL = 0
        lag = 1.01

    return dV,dL,lag


# 2.------------------------- MODEL PARAMETERS ---------------------------------
amp = 0.5                               # oscillation amplitude at base (m)
                                        # elevation change at base is twice this value

t_pd = 10*3.154e7                       # oscillation period (s)

delta = 0.1/amp                         # dimensionless displacement threshold corresponding
                                        # to dimensional threshold of 10 cm

rho = 917.0                             # ice density kg/m^3
g = 9.81                                # gravitational acceleration m^2/s
eta = 1e13                              # (constant) viscosity (Pa s)

L_true = 10*1000                        # (true) length of subglacial lake

N_pts = 20                              # number of ice thickness and friction
                                        # values (between max and min values from data)
                                        # for constructing minimum lake size function

# 3. ------------------------------ LOAD DATA-----------------------------------
gcs = gcsfs.GCSFileSystem()
H_beta_mapper = gcs.get_mapper('gs://ldeo-glaciology/bedmachine/H_beta.zarr')#, mode='ab')
H_beta = xr.open_zarr(H_beta_mapper)
H_beta.load()

X, Y = np.meshgrid(H_beta.x,H_beta.y) # map coordinates
beta_d = H_beta.beta.data             # (dimensional) friction coefficient (Pa s / m)
H = H_beta.thickness.data             # ice thickness (m)


# 4.------------COMPUTE LAKE PARAMETERS AS FUNCTION OF BETA AND H---------------

# construct arrays for H and beta_d that cover the range of the data
H_int = np.linspace(1,np.max(H),N_pts)
beta_int = np.logspace(np.min(np.log10(beta_d)),np.max(np.log10(beta_d)),N_pts)

# arrays for quantities of interest at every (H,beta_d) value
dV = np.zeros((np.size(H_int),np.size(beta_int)))       # volume change
dL = np.zeros((np.size(H_int),np.size(beta_int)))       # lake size
lag = np.zeros((np.size(H_int),np.size(beta_int)))      # phase lag

print('Computing water volume change, lake size, and timing estimates as functions of friction and ice thickness....')

l = 0
for i in range(np.shape(dV)[0]):
    for j in range(np.shape(dV)[1]):

        dV[i,j],dL[i,j],lag[i,j] = get_ratios(H_int[j],t_pd,beta_int[i],L_true)

        if l % int(np.size(dV)/10.0) == 0:
            print(str(100*l/int(np.size(dV)))+' % complete')
        l+=1

print(str(100*l/int(np.size(dV)))+' % complete')
print('\n')

# created functions from the computations above via interpolation
dV_int = interp2d(H_int,beta_int,dV,kind='linear')
dL_int = interp2d(H_int,beta_int,dL,kind='linear')
lag_int = interp2d(H_int,beta_int,lag,kind='linear')


# 5. ------------ CONSTRUCT THE MAPS BY EVALUATING THE FUNCTIONS ---------------
dV_map = np.zeros(np.shape(beta_d))     # volume change estimate map
dL_map = np.zeros(np.shape(beta_d))     # lake length estimate map
lag_map = np.zeros(np.shape(beta_d))    # phase lag map

print('Constructing map....')
l = 0
for i in range(np.shape(dV_map)[0]):
    for j in range(np.shape(dV_map)[1]):

        dV_map[i,j] = dV_int(H[i,j],beta_d[i,j])
        dL_map[i,j] = dL_int(H[i,j],beta_d[i,j])
        lag_map[i,j] = lag_int(H[i,j],beta_d[i,j])

        if l % int(np.size(dV_map)/10.0) == 0:
            print(str(100*l/int(np.size(dV_map)))+' % complete')
        l+=1

print(str(100*l/int(np.size(dV_map)))+' % complete')
print('\n')

# mask out ice shelves (friction coefficient is nonzero--but small--there)
dV_map[beta_d<1e5] = -0.01
dL_map[beta_d<1e5] = -0.01
lag_map[beta_d<1e5] = 1.01


# 6. ----------------------- PLOTTING ------------------------------------------
print('plotting....')

# customize colormaps
cmap1 = copy.copy(mpl.cm.get_cmap("Blues"))
cmap1.set_under('w')

cmap2 = copy.copy(mpl.cm.get_cmap("Reds"))
cmap2.set_under('w')

cmap3 = copy.copy(mpl.cm.get_cmap("Greens_r"))
cmap3.set_over('w')

# define contour levels for maps
levels_V = np.arange(0,1.1,step=0.2)
levels_L = np.arange(0,4.1,step=1)
levels_lag = np.arange(0,1.1,step=0.2)

fig = plt.figure(figsize=(16,6))

# volume change estimate panel
plt.subplot(131)
p1 = plt.contourf(X/1000,Y/1000,dV_map,cmap=cmap1,levels=levels_V,extend='both')
plt.contour(X/1000,Y/1000,dV_map,colors='k',levels=[0])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
cbar1 = fig.colorbar(p1,orientation='horizontal',ticks=levels_V)
cbar1.set_label(r'$ \Delta  V_\mathrm{est} \, / \, \Delta V_\mathrm{true}$',fontsize=24)
cbar1.ax.tick_params(labelsize=14)

# lake size estimate panel
plt.subplot(132)
p2 = plt.contourf(X/1000,Y/1000,dL_map,cmap=cmap2,levels=levels_L,extend='both')
plt.contour(X/1000,Y/1000,dL_map,colors='k',levels=[0])
plt.xticks(fontsize=14)
plt.gca().get_yaxis().set_ticklabels([])
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
cbar2 = fig.colorbar(p2,orientation='horizontal',ticks=levels_L)
cbar2.set_label(r'$ L_\mathrm{est} \, / \,  L_\mathrm{max}$',fontsize=24)
cbar2.ax.tick_params(labelsize=14)

# highstand/lowstand timing (phase lag) panel
plt.subplot(133)
p3 = plt.contourf(X/1000,Y/1000,lag_map,cmap=cmap3,levels=levels_lag,extend='both')
plt.contour(X/1000,Y/1000,lag_map,colors='k',levels=[1])
plt.xticks(fontsize=14)
plt.gca().get_yaxis().set_ticklabels([])
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
cbar3 = fig.colorbar(p3,orientation='horizontal',ticks=levels_lag)
cbar3.set_label(r'$\phi_\mathrm{lag}$',fontsize=24)
cbar3.ax.tick_params(labelsize=14)
cbar3.ax.invert_yaxis()

plt.tight_layout()
plt.savefig('maps')
plt.close()
