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

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

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

    T1 = np.pi - np.arctan(np.max(D2)/np.max(D1))
    T2 = 2*np.pi - np.arctan(np.max(D2)/np.max(D1))

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

    # 200 km = xp * H

    # discretization in frequency domain
    N = 2000
    x = np.linspace(-100,100,num=N)
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

    dS = 2*w                               # elevation change at base


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
        lag = (2/np.pi)*(np.pi-T1)

    else:
        dV = 0
        dL = 0
        lag = 1.01

    return dV,dL,lag

# 2.------------------------- MODEL PARAMETERS ---------------------------------
# function for spatial component of basal vertical velocity anomaly
# default is a Gaussian
def w_base(x,Ls):
    sigma = Ls/3                        # define standard deviation for Gaussian
    w = np.exp(-0.5*(x/sigma)**2)
    return w

amp = 0.5                               # oscillation amplitude at base (m)

delta = 0.1/amp                         # dimensionless displacement threshold corresponding
                                        # to dimensional threshold of 0.1 m

eta = 1e13                              # viscosity (Pa s)

rho = 917.0                             # ice density kg/m^3
g = 9.81                                # gravitational acceleration m^2/s

Ls = 10*1000.0                          # lake length (km)

N_pts = 20                              # number of ice thickness and friction
                                        # values (between max and min values from data)
                                        # for constructing minimum lake size function
                                        # (the total number of computations is N_pts**2)


# 3.---COMPUTE VOLUME CHANGE AND LAKE LENGTH RATIOS AS FUNCTIONS OF BETA AND H---

# construct arrays for H and beta_d
H = np.linspace(1000,4000,N_pts)                       # ice thickness (m)
t_pd = 3.154e7*np.linspace(1,8,N_pts)                  # oscillation period (s)

# arrays for volume change and lake length ratios at every (H,t_pd) value
dV1 = np.zeros((np.size(H),np.size(t_pd)))
dL1 = np.zeros((np.size(H),np.size(t_pd)))
lag1 = np.zeros((np.size(H),np.size(t_pd)))

dV2 = np.zeros((np.size(H),np.size(t_pd)))
dL2 = np.zeros((np.size(H),np.size(t_pd)))
lag2 = np.zeros((np.size(H),np.size(t_pd)))

dV3 = np.zeros((np.size(H),np.size(t_pd)))
dL3 = np.zeros((np.size(H),np.size(t_pd)))
lag3 = np.zeros((np.size(H),np.size(t_pd)))

dV4 = np.zeros((np.size(H),np.size(t_pd)))
dL4 = np.zeros((np.size(H),np.size(t_pd)))
lag4 = np.zeros((np.size(H),np.size(t_pd)))


print('Computing volume change and lake length ratios as functions of oscillation period and ice thickness....')

l = 0
for i in range(np.shape(dV1)[0]):
    for j in range(np.shape(dV1)[1]):
        dV1[i,j],dL1[i,j],lag1[i,j] = get_ratios(H[j],t_pd[i],1e8,Ls)
        dV2[i,j],dL2[i,j],lag2[i,j] = get_ratios(H[j],t_pd[i],1e9,Ls)
        dV3[i,j],dL3[i,j],lag3[i,j] = get_ratios(H[j],t_pd[i],1e10,Ls)
        dV4[i,j],dL4[i,j],lag4[i,j] = get_ratios(H[j],t_pd[i],1e12,Ls)


        if l % int(np.size(dV1)/10.0) == 0:
            print(str(100*l/int(np.size(dV1)))+' % complete')
        l+=1
print(str(100*l/int(np.size(dV1)))+' % complete')
print('\n')


# 4. ----------------------- PLOTTING ------------------------------------------
print('plotting....')

levelsV = np.linspace(0.0,1,num=6)
levelsL = np.linspace(0.0,4,num=5)
levels_lag = np.linspace(0.0,1,num=6)

cmap1 = copy.copy(mpl.cm.get_cmap("Blues"))
cmap1.set_under('w')

cmap2 = copy.copy(mpl.cm.get_cmap("Reds"))
cmap2.set_under('w')

cmap3 = copy.copy(mpl.cm.get_cmap("Greens_r"))
cmap3.set_over('w')

fig = plt.figure(figsize=(14,10))
plt.subplot(341)
plt.annotate(r'(a)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$\beta=10^8$ Pa s/m',fontsize=20,pad=15,bbox=dict(facecolor='seashell',alpha=1))
p1 = plt.contourf(H/1000,t_pd/3.154e7,dV1,cmap=cmap1,levels=levelsV,extend='both')
l1 = plt.contour(H/1000,t_pd/3.154e7,dV1,colors='k',linewidths=3,levels=[1e-10])
plt.ylabel(r'$T$ (yr)',fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])


plt.subplot(342)
plt.annotate(r'(b)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$\beta=10^9$ Pa s/m',fontsize=20,pad=15,bbox=dict(facecolor='seashell',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dV2,cmap=cmap1,levels=levelsV,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dV2,colors='k',linewidths=3,levels=[1e-10])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])


plt.subplot(343)
plt.annotate(r'(c)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$\beta=10^{10}$ Pa s/m',fontsize=20,pad=15,bbox=dict(facecolor='seashell',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dV3,cmap=cmap1,levels=levelsV,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dV3,colors='k',linewidths=3,levels=[1e-10])
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])



plt.subplot(344)
plt.annotate(r'(d)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$\beta=10^{12}$ Pa s/m',fontsize=20,pad=15,bbox=dict(facecolor='seashell',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dV4,cmap=cmap1,levels=levelsV,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dV4,colors='k',linewidths=3,levels=[1e-10])
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.875, 0.665, 0.02, 0.2])
cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levelsV)
cbar.set_label(r'$\frac{\Delta  V_\mathrm{est}}{\Delta V_\mathrm{true}}$',verticalalignment='center', rotation=0,fontsize=30)
cbar.ax.set_yticklabels(['0.0 (not detected)','0.2','0.4','0.6','0.8','1.0'])
cbar.ax.get_yaxis().labelpad = -70
cbar.ax.tick_params(labelsize=16)

cbar.add_lines(l1)


#------------------- length estimates-------------------------------------------
plt.subplot(345)
plt.annotate(r'(e)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
p2 = plt.contourf(H/1000,t_pd/3.154e7,dL1,cmap=cmap2,levels=levelsL,extend='both')
l2 = plt.contour(H/1000,t_pd/3.154e7,dL1,colors='k',linewidths=3,levels=[1e-10])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.ylabel(r'$T$ (yr)',fontsize=20)

plt.subplot(346)
plt.annotate(r'(f)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dL2,cmap=cmap2,levels=levelsL,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dL2,colors='k',linewidths=3,levels=[1e-10])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])


plt.subplot(347)
plt.annotate(r'(g)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dL3,cmap=cmap2,levels=levelsL,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dL3,colors='k',linewidths=3,levels=[1e-10])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])

plt.subplot(348)
plt.annotate(r'(h)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,dL4,cmap=cmap2,levels=levelsL,extend='both')
plt.contour(H/1000,t_pd/3.154e7,dL4,colors='k',linewidths=3,levels=[1e-10])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.gca().yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.875, 0.3925, 0.02, 0.2])
cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levelsL)
cbar.set_label(r'$\frac{L_\mathrm{est}}{L_\mathrm{max}}$',verticalalignment='center', rotation=0,fontsize=30)
cbar.ax.get_yaxis().labelpad = -80
cbar.ax.tick_params(labelsize=16)
cbar.ax.set_yticklabels(['0.0 (not detected)','1','2','3','4'])
cbar.add_lines(l2)

#------------------- length estimates-------------------------------------------

plt.subplot(349)
plt.annotate(r'(i)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
p3 = plt.contourf(H/1000,t_pd/3.154e7,lag1,cmap=cmap3,levels=levels_lag,extend='both')
l3 = plt.contour(H/1000,t_pd/3.154e7,lag1,colors='k',linewidths=3,levels=[1])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'$H$ (km)',fontsize=20)
plt.ylabel(r'$T$ (yr)',fontsize=20)


plt.subplot(3,4,10)
plt.annotate(r'(j)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,lag2,cmap=cmap3,levels=levels_lag,extend='both')
plt.contour(H/1000,t_pd/3.154e7,lag2,colors='k',linewidths=3,levels=[1])
plt.xticks(fontsize=16)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel(r'$H$ (km)',fontsize=20)


plt.subplot(3,4,11)
plt.annotate(r'(k)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,lag3,cmap=cmap3,levels=levels_lag,extend='both')
plt.contour(H/1000,t_pd/3.154e7,lag3,colors='k',linewidths=3,levels=[1])
plt.xticks(fontsize=16)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel(r'$H$ (km)',fontsize=20)


plt.subplot(3,4,12)
plt.annotate(r'(l)',xy=(1.075,7.325),fontsize=16,bbox=dict(facecolor='w',alpha=1))
plt.contourf(H/1000,t_pd/3.154e7,lag4,cmap=cmap3,levels=levels_lag,extend='both')
plt.contour(H/1000,t_pd/3.154e7,lag4,colors='k',linewidths=3,levels=[1])
plt.xticks(fontsize=16)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel(r'$H$ (km)',fontsize=20)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.875, 0.12, 0.02, 0.2])
cbar = fig.colorbar(p3,cax=cbar_ax,orientation='vertical',ticks=levels_lag)
cbar.set_label(r'$\phi_\mathrm{lag}$',verticalalignment='center', rotation=0,fontsize=24)
cbar.ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0 (not detected)'])
cbar.ax.get_yaxis().labelpad = -70
cbar.ax.tick_params(labelsize=16)
cbar.ax.invert_yaxis()
cbar.add_lines(l3)

plt.savefig('ratios',bbox_inches='tight')
plt.close()
