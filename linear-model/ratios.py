# ratios.py: simple method for estimating volume change and lake length ratios
#
# OVERVIEW
# this code constructs plots of estimated vs. true subglacial water volume change and
# subglacial lake length over a range of ice thicknesses and oscillation periods.
# the computation is based on a small-perturbation ice-flow model
# --see the supporting information for a full description of the method.
#
# the main parameters that can be set below are:
#  (1) the (dimensional) basal friction coefficient beta_d)
#  (2) the subglacial lake length (Ls)
#  (3) the spatial component of the lake's basal vertical velocity anomaly (w_base); default is a Gaussian
#  (4) the maximum amplitude of the oscillation (amp)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,interpolate
from scipy import integrate
from scipy.fft import fft,ifft,fftshift,fftfreq
import matplotlib as mpl
import numpy as np
from scipy.io import loadmat
import copy


# 1.---------------FUNCTIONS FOR VOLUME CHANGE / LENGTH RATIOS------------------
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
    # Times where elevation anomaly is maximized (T1) and minimized (T2),
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

def get_ratios(H,t_pd,beta_d,Ls):
    # compute ratios of the estimated lake length (dL) and water volume change (dV)
    # relative to their true values given the true lake length (Ls),
    # dimensional friction (beta_d), and ice thickness (H)

    # discretization in frequency domain
    N = 2000
    x = np.linspace(-200,200,num=N)
    d = np.abs(x[1]-x[0])
    k = fftfreq(N,d)   # frequency
    k[0] = 1e-10       # set zero frequency to small number due to (integrable) singularity
    k *= 2*np.pi       # convert to SciPy's Fourier transform definition (angular
                       # freq. definition) used in notes

    w = w_base(x,Ls/H)                     # compute basal velocity anomaly

    w_ft = fft(w)                          # fourier transform for numerical method

    beta_nd = beta_d*H/(2*eta)             # non-dimensional friction parameter
                                           # relative to viscosity/ice thickness

    tr =  (4*np.pi*eta)/(rho*g*H)          # relaxation time
    lamda = t_pd/tr                        # ratio of oscillation time to relaxation time

    D1,D2 = get_Dj(lamda,beta_nd,w_ft,k)   # compute surface displacements

    T1,T2 = get_Tj(D1,D2,x)                # compute estimated highstand/lowstand times

    kappa1,kappa2 = get_kappaj(T1,T2)      # compute weights for displacements

    dH = kappa1*D1 + kappa2*D2             # compute surface elevation change anomaly

    dS = kappa1*w

    # interpolate displacements for integration
    dSi = interpolate.interp1d(x, dS,fill_value="extrapolate")
    dHi = interpolate.interp1d(x, dH,fill_value="extrapolate")

    dVs = integrate.quad(dSi,-0.5*Ls/H,0.5*Ls/H,full_output=1)[0]

    # compute estimated lake length
    if np.size(x[np.abs(dH)>delta])>0:
        x0 = x[np.abs(dH)>delta]
    else:
        x0 = 0*x

    Lh = 2*np.max(x0)                 # (problem is symmetric with respect to x)

    if Lh > 1e-5:
        dVh = integrate.quad(dHi,-0.5*Lh,0.5*Lh,full_output=1)[0]
        dV = dVh/dVs
        dL = Lh*H/Ls
    else:
        dV = 0
        dL = 0

    return dV,dL

# 2.------------------------- MODEL PARAMETERS ---------------------------------
# function for spatial component of basal vertical velocity anomaly
# default is a Gaussian
def w_base(x,Ls):
    sigma = Ls/6                        # define standard deviation for Gaussian
    w = np.exp(-0.5*(x/sigma)**2)
    return w

amp = 1                                 # oscillation amplitude at base (m)

delta = 0.1/amp                         # dimensionless displacement threshold corresponding
                                        # to dimensional threshold of 0.1 m

eta = 1e12                              # constant viscosity

beta_d = 1e10                           # (dimensional) basal friction coefficient (Pa s/m)


rho = 917.0                             # ice density kg/m^3
g = 9.81                                # gravitational acceleration m^2/s

Ls = 12*1000.0                          # lake length (relative to ice thickness)

N_pts = 20                              # number of ice thickness and friction
                                        # values (between max and min values from data)
                                        # for constructing minimum lake size function
                                        # (the total number of computations is N_pts**2)


# 3.---COMPUTE VOLUME CHANGE AND LAKE LENGTH RATIOS AS FUNCTIONS OF BETA AND H---

# construct arrays for H and beta_d
H = np.linspace(1000,4000,N_pts)                       # ice thickness (m)
t_pd = 3.154e7*np.linspace(1,8,N_pts)                  # oscillation period (s)

# arrays for volume change and lake length ratios at every (H,t_pd) value
dV = np.zeros((np.size(H),np.size(t_pd)))
dL = np.zeros((np.size(H),np.size(t_pd)))


print('Computing volume change and lake length ratios as functions of oscillation period and ice thickness....')

l = 0
for i in range(np.shape(dV)[0]):
    for j in range(np.shape(dV)[1]):
        dV[i,j],dL[i,j] = get_ratios(H[j],t_pd[i],beta_d,Ls)
        if l % int(np.size(dV)/10.0) == 0:
            print(str(100*l/int(np.size(dV)))+' % complete')
        l+=1
print(str(100*l/int(np.size(dV)))+' % complete')
print('\n')


# 4. ----------------------- PLOTTING ------------------------------------------
print('plotting....')

levelsV = np.linspace(0.0,1,num=6)
levelsL = np.linspace(0.0,3,num=7)

cmap1 = copy.copy(mpl.cm.get_cmap("Blues"))
cmap1.set_under('w')

cmap2 = copy.copy(mpl.cm.get_cmap("Reds"))
cmap2.set_under('w')

fig = plt.figure(figsize=(6,6))
plt.suptitle(r'$\mathrm{log}(\beta)=$'+str(int(np.log10(beta_d))),fontsize=24)
plt.subplot(211)
p1 = plt.contourf(H/1000,t_pd/3.154e7,dV,cmap=cmap1,levels=levelsV,extend='both')
l1 = plt.contour(H/1000,t_pd/3.154e7,dV,colors='k',linewidths=3,levels=[1e-10])
plt.ylabel(r'$T$ (yr)',fontsize=20)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])

cbar = fig.colorbar(p1,orientation='vertical',ticks=levelsV,aspect=15)
cbar.set_label(r'${\Delta  V_\mathrm{est}}\,/\,{\Delta V_\mathrm{true}}$',verticalalignment='center',fontsize=20)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.tick_params(labelsize=16)
cbar.add_lines(l1)

plt.subplot(212)
p2 = plt.contourf(H/1000,t_pd/3.154e7,dL,cmap=cmap2,levels=levelsL,extend='both')
l2 = plt.contour(H/1000,t_pd/3.154e7,dL,colors='k',linewidths=3,levels=[1e-10])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'$H$ (km)',fontsize=20)
plt.ylabel(r'$T$ (yr)',fontsize=20)

cbar = fig.colorbar(p2,orientation='vertical',ticks=levelsL,aspect=15)
cbar.set_label(r'${L_\mathrm{est}}\,/\,{L_\mathrm{true}}$',verticalalignment='center',fontsize=20)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.tick_params(labelsize=16)
cbar.add_lines(l2)

plt.tight_layout()
plt.savefig('ratios')
