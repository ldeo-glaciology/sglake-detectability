# map.py: simple method for estimating minimum detectable subglacial lake size
#
# OVERVIEW
# this code constructs a map of the minimum detectable subglacial lake size
# given maps of the ice thickness and basal friction coefficient.
# the computation is based on a small-perturbation ice-flow model
# --see the supporting information for a full description of the method.
#
# the main parameters that can be set below are:
#  (1) the subglacial lake oscillation period, t_pd, (sinusoidal in time),
#  (2) the spatial component of the lake's basal vertical velocity anomaly, w_base; (default is a Gaussian),
#  (3) the maximum amplitude of the oscillation, amp
#
# DATA
# the ice thickness and basal friction maps used here (by default) were
# obtained by personal communication between J. Kingslake (Columbia University)
# and R. Arthern (British Antarctic Survey), and featured in the publication:
#
#   Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within
#   the Antarctic ice sheet and its controls inferred from satellite observations.
#   Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.
#
# (other maps can be used if they can be imported as NumPy arrays)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.fft import fft,ifft,fftshift,fftfreq
import matplotlib as mpl
import numpy as np
from scipy.io import loadmat
import copy


# 1.---------------FUNCTIONS FOR COMPUTING MINIMUM DETECTABLE LAKE SIZE---------
def get_Dj(lamda,beta_nd,w_ft,k):
    # function for computing displacements D1 (in-phase with base) and D2 (anti-phase with base)
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
    # Times where elevation anomaly is maximized (T1) and minimized (T2)
    i0 = np.argmin(np.abs(x))

    T1 = np.pi - np.arctan(D2[i0]/D1[i0])
    T2 = 2*np.pi - np.arctan(D2[i0]/D1[i0])

    return T1,T2

def get_kappaj(T1,T2):
    # weights on the displacements:
    # kappa1 is the in-phase component
    # kappa2 is the anti-phase component
    kappa1 = np.cos(T2) - np.cos(T1)
    kappa2 = np.sin(T1) - np.sin(T2)

    return kappa1,kappa2

def get_Lh(Ls,beta_d,H):
    # compute the estimated length of the lake (Lh) given the true lake length (Ls),
    # dimensional friction (beta_d), and ice thickness (H)

    # discretization in frequency domain
    N = 2000
    x = np.linspace(-200,200,num=N)
    d = np.abs(x[1]-x[0])
    k = fftfreq(N,d)   # frequency
    k[0] = 1e-10       # set zero frequency to small number due to (integrable) singularity
    k *= 2*np.pi       # convert to SciPy's Fourier transform definition from version (angular
                       # freq. definition) used in model derivation

    w = w_base(x,Ls)                       # compute basal velocity anomaly

    w_ft = fft(w)                          # fourier transform (for numerical method)

    beta_nd = beta_d*H/(2*eta)             # non-dimensional friction parameter
                                           # relative to viscosity/ice thickness

    tr =  (4*np.pi*eta)/(rho*g*H)          # relaxation time
    lamda = t_pd/tr                        # ratio of oscillation time to relaxation time


    D1,D2 = get_Dj(lamda,beta_nd,w_ft,k)   # compute surface displacements

    T1,T2 = get_Tj(D1,D2,x)                # compute estimated highstand/lowstand times

    kappa1,kappa2 = get_kappaj(T1,T2)      # compute weights for displacements

    dH = kappa1*D1 + kappa2*D2             # compute surface elevation change anomaly

    # compute estimated lake length
    if np.size(x[np.abs(dH)>delta])>0:
        x0 = x[np.abs(dH)>delta]
    else:
        x0 = 0*x

    Lh = 2*np.max(x0)                 # (problem is symmetric with respect to x)

    return Lh

def get_min_Ls(Ls,beta_d,H):
    # compute the minimum detectable lake size by starting with a huge lake size
    # and gradually decreasing it until the estimated lake length vanishes

    Ls_min = Ls[-1]                 # initialize as smallest lake size in case
                                    # all lake sizes are detectable

    for i in range(np.size(Ls)):
        Lh = get_Lh(Ls[i],beta_d,H)

        if Lh < 1e-5:
            Ls_min = Ls[i]
            break

    return Ls_min

# 2.------------------------- MODEL PARAMETERS ---------------------------------

# function for spatial component of basal vertical velocity anomaly
# default is a Gaussian
def w_base(x,Ls):
    sigma = Ls/6                        # define standard deviation for Gaussian
    w = np.exp(-0.5*(x/sigma)**2)
    return w

amp = 1                                 # oscillation amplitude at base (m)
t_pd = 10*3.154e7                       # oscillation period (s)

delta = 0.1/amp                         # dimensionless displacement threshold corresponding
                                        # to dimensional threshold of 0.1 m

rho = 917.0                             # ice density kg/m^3
g = 9.81                                # gravitational acceleration m^2/s
eta = 1e12                              # constant viscosity (Pa s)

Ls = np.linspace(100,1,101)             # array of lake lengths (relative to ice thickness)
                                        # for computing the minimum detectable lake size
                                        # default minmium value = 1, maximum value = 100.

N_pts = 20                              # number of ice thickness and friction
                                        # values (between max and min values from data)
                                        # for constructing minimum lake size function

# 3. ------------------------------ LOAD DATA-----------------------------------
data_dict = loadmat('beta_h.mat')

X = data_dict['x']                      # horizontal x coordinate
Y = data_dict['y']                      # horizontal y coordinate
beta_d = data_dict['beta']*3.154e7      # (dimensional) friction coefficient (Pa s / m)
H = data_dict['h']                      # ice thickness (m)


# 4.------COMPUTE MINIMUM DETECTABLE LAKE SIZE AS FUNCTION OF BETA AND H--------

# construct arrays for H and beta_d that cover the range of the data
H_int = np.linspace(1,np.max(H),N_pts)
beta_int = np.logspace(np.min(np.log10(beta_d)),np.max(np.log10(beta_d)),N_pts)

# array for minimum lake size at every (H,beta_d) value
min_Ls = np.zeros((np.size(H_int),np.size(beta_int)))

print('Computing minimum detectable lake size as function of friction and ice thickness....')

l = 0
for i in range(np.shape(min_Ls)[0]):
    for j in range(np.shape(min_Ls)[1]):
        min_Ls[i,j] = get_min_Ls(Ls,beta_int[i],H_int[j])
        if l % int(np.size(min_Ls)/10.0) == 0:
            print(str(100*l/int(np.size(min_Ls)))+' % complete')
        l+=1
print(str(100*l/int(np.size(min_Ls)))+' % complete')
print('\n')

MLS = interp2d(H_int,beta_int,min_Ls,kind='linear')

# 5. ------------ CONSTRUCT THE MAP BY EVALUATING THE FUNCTION -----------------
Ls_map = np.zeros(np.shape(beta_d))     # minimum lake length map

print('Constructing map....')
l = 0
for i in range(np.shape(Ls_map)[0]):
    for j in range(np.shape(Ls_map)[1]):
        Ls_map[i,j] = MLS(H[i,j],beta_d[i,j])
        if l % int(np.size(Ls_map)/10.0) == 0:
            print(str(100*l/int(np.size(Ls_map)))+' % complete')
        l+=1
print(str(100*l/int(np.size(Ls_map)))+' % complete')
print('\n')


# 6. ----------------------- PLOTTING ------------------------------------------
print('plotting....')
levels_H = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500])/1000.0
levels_beta = np.array([1e6,1e8,1e10,1e12,1e14,1e16])

levels_L = np.array([0,5,10,15,20,25,30])


cmap1 = copy.copy(mpl.cm.get_cmap("Blues"))
cmap1.set_under('mistyrose')

plt.figure(figsize=(14,6))
plt.suptitle(r'$t_p$ = '+"{:.1f}".format(t_pd/3.154e7)+r' yr,   $\mathcal{A} =$ '+"{:.1f}".format(amp)+' m',fontsize=24)
plt.subplot(131)
p1 = plt.contourf(X/1000,Y/1000,Ls_map*H/1000.0,cmap=cmap1,levels=levels_L,extend='both')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y$',fontsize=16)
cbar = plt.colorbar(p1,orientation='horizontal')
cbar.set_label(r'$L_\mathrm{min}$ (km)',fontsize=20)


plt.subplot(132)
plt.xticks(fontsize=12)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel(r'$x$',fontsize=16)
p1 = plt.contourf(X/1000,Y/1000,H/1000.0,cmap=cmap1,levels=levels_H,extend='both')
cbar = plt.colorbar(p1,orientation='horizontal')
cbar.set_label(r'$H$ (km)',fontsize=20)

plt.subplot(133)
plt.xticks(fontsize=12)
plt.xlabel(r'$x$',fontsize=16)
plt.gca().yaxis.set_ticklabels([])
p2 = plt.contourf(X/1000,Y/1000,beta_d,cmap=cmap1,levels=levels_beta,norm=mpl.colors.LogNorm(),extend='both')
cbar = plt.colorbar(p2,orientation='horizontal')
cbar.set_label(r'$\beta$ (Pa s / m)',fontsize=20)

plt.tight_layout()
plt.savefig('maps')
plt.close()
