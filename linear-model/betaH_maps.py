# OVERVIEW
# this code plots the following data set:

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
import matplotlib as mpl
import numpy as np
import copy
import xarray as xr
import matplotlib.colors as colors



# 3. ------------------------------ LOAD DATA-----------------------------------
H_beta = xr.open_zarr('data/H_beta.zarr')
H_beta.load()

X, Y = np.meshgrid(H_beta.x,H_beta.y) # horizontal map coordinates
beta_d = H_beta.beta.data               # (dimensional) basal sliding coefficient (Pa s / m)
H = H_beta.thickness.data               # (dimensional) basal sliding coefficient (Pa s / m)



# 4. ----------------------- PLOTTING ------------------------------------------
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


#------------------ Col 1------------------------------------------
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.annotate(r'(a)',xy=(-2740,2540),fontsize=18,bbox=dict(facecolor='w',alpha=1))
p1 = plt.contourf(X/1000,Y/1000,H/1000,cmap='Greys',vmin=0,vmax=np.max(H/1000),levels=[0,1,2,3,4],extend='both')
plt.contour(X/1000,Y/1000,H/1000,colors='k',levels=[0])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.xlabel(r'$x$ (km)',fontsize=20)
cbar = plt.colorbar(p1,ticks=[0,1,2,3,4],orientation='horizontal')
cbar.set_label(r'$H$ (km)', fontsize=20)
cbar.ax.tick_params(labelsize=14)

plt.subplot(122)
plt.annotate(r'(b)',xy=(-2740,2540),fontsize=18,bbox=dict(facecolor='w',alpha=1))
p2 = plt.contourf(X/1000,Y/1000,beta_d,cmap='Greys',norm=colors.LogNorm(vmin=1e6, vmax=1e14),levels=[1e6,1e8,1e10,1e12,1e14],extend='both')
plt.contour(X/1000,Y/1000,H/1000,colors='k',levels=[0])
plt.xticks(fontsize=14)
plt.gca().get_yaxis().set_ticklabels([])
plt.xlabel(r'$x$ (km)',fontsize=20)
cbar = plt.colorbar(p2,orientation='horizontal')
cbar.set_label(r'$\beta$ (Pa s m$^{-1})$', fontsize=20)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()

plt.savefig('fig_S1',bbox_inches='tight')
plt.close()
