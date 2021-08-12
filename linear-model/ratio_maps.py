# ratio_maps.py: code for computing subglacial lake size, volume change, and phase lag
#         discrepancies across the Antarctic ice sheet
#
# OVERVIEW
# this code constructs a maps of subglacial lake size, volume change, and phase lag
# discrepancies given maps of the ice thickness and basal sliding coefficient.
# the computation is based on a small-perturbation ice-flow model
# --see the supporting information for a full description of the method.
#
# the main parameters that can be set below are:
#  (1) the subglacial lake oscillation period, t_pd, (sinusoidal in time),
#  (2) the spatial component of the lake's basal vertical velocity anomaly, w_base; (default is a Gaussian),
#  (3) the maximum amplitude of the oscillation, amp
#
# DATA
# the ice thickness and basal sliding coefficient maps used here (by default) were
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
from scipy.interpolate import interp2d,interpolate
from scipy import integrate
from scipy.fft import fft,ifft,fftshift,fftfreq
import matplotlib as mpl
import numpy as np
import copy
import xarray as xr


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

def get_Tj(D1,D2,x,H):
    # Times where elevation anomaly is maximized (T1) and minimized (T2)
    i0 = np.argmin(np.abs(x))

    T1 = np.pi - np.arctan(np.mean(D2[np.abs(x)*H/1000<10])/np.mean(D1[np.abs(x)*H/1000<10]))
    T2 = 2*np.pi - np.arctan(np.mean(D2[np.abs(x)*H/1000<10])/np.mean(D1[np.abs(x)*H/1000<10]))

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
    # dimensional drag (beta_d), and ice thickness (H)

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

    beta_nd = beta_d*H/(2*eta)             # non-dimensional basal drag parameter
                                           # relative to viscosity/ice thickness

    tr =  (4*np.pi*eta)/(rho*g*H)          # relaxation time
    lamda = t_pd/tr                        # ratio of oscillation time to relaxation time

    D1,D2 = get_Dj(lamda,beta_nd,w_ft,k)   # compute surface displacements

    T1,T2 = get_Tj(D1,D2,x,H)              # compute estimated highstand/lowstand times

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
    sigma = Ls/4                        # define standard deviation for Gaussian
    w = np.exp(-0.5*(x/sigma)**2)
    return w

amp = 0.5                               # oscillation amplitude at base (m)
                                        # elevation change at base is twice this value

t_pd = 5*3.154e7                        # oscillation period (s)

delta = 0.1/amp                         # dimensionless displacement threshold corresponding
                                        # to dimensional threshold of 0.1 m

rho = 917.0                             # ice density kg/m^3
g = 9.81                                # gravitational acceleration m^2/s
eta = 1e13                              # constant viscosity (Pa s)

Ls = 10*1000                            # array of lake lengths (relative to ice thickness)
                                        # for computing the minimum detectable lake size
                                        # default minmium value = 1, maximum value = 100.

N_pts = 20                             # number of ice thickness and basal drag
                                        # values (between max and min values from data)
                                        # for constructing minimum lake size function

# 3. ------------------------------ LOAD DATA-----------------------------------
# Load thickness and basal sliding coefficient
H_beta = xr.open_zarr('data/H_beta.zarr')
H_beta.load()

X, Y = np.meshgrid(H_beta.x,H_beta.y)   # horizontal map coordinates
beta_d = H_beta.beta.data               # (dimensional) drag coefficient (Pa s / m)
H = H_beta.thickness.data               # ice thickness (m)

active = np.loadtxt('data/active_lake_statistics.dat',delimiter=',')

# lake coordinates
x_act = active[:,0]
y_act = active[:,1]

# lake length estimates (km)
act_l2 = 1000*active[:,3]



# 4.------COMPUTE MINIMUM DETECTABLE LAKE SIZE AS FUNCTION OF BETA AND H--------

# construct arrays for H and beta_d that cover the range of the data
H_int = np.linspace(1,np.max(H),N_pts)
beta_int = np.logspace(np.min(np.log10(beta_d)),np.max(np.log10(beta_d)),N_pts)

# array for minimum lake size at every (H,beta_d) value
dV = np.zeros((np.size(H_int),np.size(beta_int)))
dL = np.zeros((np.size(H_int),np.size(beta_int)))
lag = np.zeros((np.size(H_int),np.size(beta_int)))

dV2 = np.zeros((np.size(H_int),np.size(beta_int)))
dL2 = np.zeros((np.size(H_int),np.size(beta_int)))
lag2 = np.zeros((np.size(H_int),np.size(beta_int)))

dV3 = np.zeros((np.size(H_int),np.size(beta_int)))
dL3 = np.zeros((np.size(H_int),np.size(beta_int)))
lag3 = np.zeros((np.size(H_int),np.size(beta_int)))

print('Computing discrepancies as function of basal sliding coefficient and ice thickness....')

l = 0
for i in range(np.shape(dV)[0]):
    for j in range(np.shape(dV)[1]):

        dV[i,j],dL[i,j],lag[i,j] = get_ratios(H_int[j],t_pd,beta_int[i],Ls)
        dV2[i,j],dL2[i,j],lag2[i,j] = get_ratios(H_int[j],2*t_pd,beta_int[i],Ls)
        dV3[i,j],dL3[i,j],lag3[i,j] = get_ratios(H_int[j],4*t_pd,beta_int[i],Ls)

        if l % int(np.size(dV)/10.0) == 0:
            print(str(100*l/int(np.size(dV)))+' % complete')
        l+=1

print(str(100*l/int(np.size(dV)))+' % complete')
print('\n')

dV_int = interp2d(H_int,beta_int,dV,kind='linear')
dL_int = interp2d(H_int,beta_int,dL,kind='linear')
lag_int = interp2d(H_int,beta_int,lag,kind='linear')

dV2_int = interp2d(H_int,beta_int,dV2,kind='linear')
dL2_int = interp2d(H_int,beta_int,dL2,kind='linear')
lag2_int = interp2d(H_int,beta_int,lag2,kind='linear')

dV3_int = interp2d(H_int,beta_int,dV3,kind='linear')
dL3_int = interp2d(H_int,beta_int,dL3,kind='linear')
lag3_int = interp2d(H_int,beta_int,lag3,kind='linear')

# 5. ------------ CONSTRUCT THE MAP BY EVALUATING THE FUNCTION -----------------
dV_map = np.zeros(np.shape(beta_d))     # minimum lake length map
dL_map = np.zeros(np.shape(beta_d))     # example estimated lake length map
lag_map = np.zeros(np.shape(beta_d))     # example estimated lake length map

dV2_map = np.zeros(np.shape(beta_d))     # minimum lake length map
dL2_map = np.zeros(np.shape(beta_d))     # example estimated lake length map
lag2_map = np.zeros(np.shape(beta_d))     # example estimated lake length map

dV3_map = np.zeros(np.shape(beta_d))     # minimum lake length map
dL3_map = np.zeros(np.shape(beta_d))     # example estimated lake length map
lag3_map = np.zeros(np.shape(beta_d))     # example estimated lake length map

print('Constructing map....')
l = 0
for i in range(np.shape(dV_map)[0]):
    for j in range(np.shape(dV_map)[1]):

        dV_map[i,j] = dV_int(H[i,j],beta_d[i,j])
        dL_map[i,j] = dL_int(H[i,j],beta_d[i,j])
        lag_map[i,j] = lag_int(H[i,j],beta_d[i,j])

        dV2_map[i,j] = dV2_int(H[i,j],beta_d[i,j])
        dL2_map[i,j] = dL2_int(H[i,j],beta_d[i,j])
        lag2_map[i,j] = lag2_int(H[i,j],beta_d[i,j])

        dV3_map[i,j] = dV3_int(H[i,j],beta_d[i,j])
        dL3_map[i,j] = dL3_int(H[i,j],beta_d[i,j])
        lag3_map[i,j] = lag3_int(H[i,j],beta_d[i,j])

        if l % int(np.size(dV_map)/10.0) == 0:
            print(str(100*l/int(np.size(dV_map)))+' % complete')
        l+=1
print(str(100*l/int(np.size(dV_map)))+' % complete')
print('\n')

# mask out ice shelves
dV_map[beta_d<1e5] = 0
dL_map[beta_d<1e5] = 0
lag_map[beta_d<1e5] = 1.01

dV2_map[beta_d<1e5] = 0
dL2_map[beta_d<1e5] = 0
lag2_map[beta_d<1e5] = 1.01

dV3_map[beta_d<1e5] = 0
dL3_map[beta_d<1e5] = 0
lag3_map[beta_d<1e5] = 1.01

# 6. ----------------------- PLOTTING ------------------------------------------
print('plotting....')


# #MAP W/ LAKES
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

levels_V = np.arange(0,1.1,step=0.2)
levels_L = np.linspace(0.0,3,num=6)
levels_lag = np.arange(0,1.1,step=0.2)


cmap1 = copy.copy(mpl.cm.get_cmap("Blues"))
cmap1.set_under('w')

cmap2 = copy.copy(mpl.cm.get_cmap("Reds"))
cmap2.set_under('w')

cmap3 = copy.copy(mpl.cm.get_cmap("Greens_r"))
cmap3.set_over('w')

fig = plt.figure(figsize=(16,13))


#------------------------------- Col 1------------------------------------------
plt.subplot(331)
plt.annotate(r'(a)',xy=(-2720,2420),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$T=5$ yr',fontsize=24,pad=15,bbox=dict(facecolor='seashell',alpha=1))
l1 = plt.contour(X/1000,Y/1000,dV_map,colors='k',levels=[0])
p1 = plt.contourf(X/1000,Y/1000,dV_map,cmap=cmap1,levels=levels_V,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')

plt.gca().get_xaxis().set_ticklabels([])
plt.yticks(fontsize=14)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(334)
plt.annotate(r'(d)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.contour(X/1000,Y/1000,dV_map,colors='k',levels=[0])
p2 = plt.contourf(X/1000,Y/1000,dL_map,cmap=cmap2,levels=levels_L,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().set_aspect('equal', adjustable='box')


plt.subplot(337)
plt.annotate(r'(g)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
l3 = plt.contour(X/1000,Y/1000,dV_map,colors='k',levels=[0])
p3 = plt.contourf(X/1000,Y/1000,lag_map,cmap=cmap3,levels=levels_lag,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')


plt.tight_layout()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.86, 0.71, 0.02, 0.2])
cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels_V)
cbar.set_label(r'$\frac{\Delta  V_\mathrm{est}}{\Delta V_\mathrm{true}}$',verticalalignment='center', rotation=0,fontsize=30)
cbar.ax.set_yticklabels(['0.0 (unobservable)','0.2','0.4','0.6','0.8','1.0'])
cbar.ax.get_yaxis().labelpad = -75
cbar.ax.tick_params(labelsize=16)
cbar.add_lines(l1)


#------------------------------- Col 2------------------------------------------
plt.subplot(332)
plt.annotate(r'(b)',xy=(-2720,2420),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$T=10$ yr',fontsize=24,pad=15,bbox=dict(facecolor='seashell',alpha=1))
plt.contour(X/1000,Y/1000,dV2_map,colors='k',levels=[0])
p1 = plt.contourf(X/1000,Y/1000,dV2_map,cmap=cmap1,levels=levels_V,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')


plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(335)
plt.annotate(r'(e)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
l2 = plt.contour(X/1000,Y/1000,dL2_map,colors='k',levels=[0])
p2 = plt.contourf(X/1000,Y/1000,dL2_map,cmap=cmap2,levels=levels_L,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')

plt.gca().get_yaxis().set_ticklabels([])
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().set_aspect('equal', adjustable='box')


plt.subplot(338)
plt.annotate(r'(h)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.contour(X/1000,Y/1000,dV2_map,colors='k',levels=[0])
plt.contourf(X/1000,Y/1000,lag2_map,cmap=cmap3,levels=levels_lag,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')
plt.gca().get_yaxis().set_ticklabels([])
plt.xticks(fontsize=14)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.86, 0.4, 0.02, 0.2])
cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels_L)
cbar.set_label(r'$\frac{L_\mathrm{est}}{L_\mathrm{true}}$',verticalalignment='center', rotation=0,fontsize=30)
cbar.ax.set_yticklabels(['0.0 (unobservable)','1','2','3','4','5'])
cbar.ax.get_yaxis().labelpad = -75
cbar.ax.tick_params(labelsize=16)
cbar.add_lines(l2)


#------------------------------- COL 3------------------------------------------
plt.subplot(333)
plt.annotate(r'(c)',xy=(-2720,2420),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.title(r'$T=20$ yr',fontsize=24,pad=15,bbox=dict(facecolor='seashell',alpha=1))
plt.contour(X/1000,Y/1000,dV3_map,colors='k',levels=[0])
p1 = plt.contourf(X/1000,Y/1000,dV3_map,cmap=cmap1,levels=levels_V,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')

plt.gca().get_yaxis().set_ticklabels([])
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(336)
plt.annotate(r'(f)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
plt.contour(X/1000,Y/1000,dV3_map,colors='k',levels=[0])
p2 = plt.contourf(X/1000,Y/1000,dL3_map,cmap=cmap2,levels=levels_L,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')

plt.xticks(fontsize=16)
plt.gca().get_yaxis().set_ticklabels([])
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().set_aspect('equal', adjustable='box')


plt.subplot(339)
plt.annotate(r'(i)',xy=(-2720,2440),fontsize=18,bbox=dict(facecolor='w',alpha=1))
l3 = plt.contour(X/1000,Y/1000,lag3_map,colors='k',levels=[1])
p3 = plt.contourf(X/1000,Y/1000,lag3_map,cmap=cmap3,levels=levels_lag,extend='both')
sc = plt.scatter(x=x_act/1000,y=y_act/1000,s=act_l2/1000,edgecolors='yellow',facecolors='none',linewidths=1.5,marker='o')
plt.xticks(fontsize=14)
plt.gca().get_yaxis().set_ticklabels([])
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')


plt.tight_layout()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.86, 0.105, 0.02, 0.2])
cbar = fig.colorbar(p3,cax=cbar_ax,orientation='vertical',ticks=levels_lag)
cbar.set_label(r'$\phi_\mathrm{lag}$',verticalalignment='center', rotation=0,fontsize=24)
cbar.ax.get_yaxis().labelpad = -75
cbar.ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0 (unobservable)'])
cbar.ax.tick_params(labelsize=16)
cbar.ax.invert_yaxis()
cbar.add_lines(l3)

kw = dict(prop='sizes',color='none',num=6,markeredgecolor='yellow')
lgd = plt.legend(*sc.legend_elements(**kw),prop={'size':16},bbox_to_anchor=(-5,-0.5),edgecolor='k',framealpha=1,ncol=7,facecolor='cornflowerblue')
lgd.set_title(r'$L_\mathrm{est}$ (km)',prop={'size':20})
#plt.setp(lgd.get_texts(), color='w')

plt.savefig('fig3',bbox_inches='tight')
plt.close()
