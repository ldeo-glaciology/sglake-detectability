#-------------------------------------------------------------------------------
# This function defines the rate of water volume change in the subglacial lake.
# *Default volume change = sinusoidal timeseries
# (of course you can calculate dVol/dt analytically for the default example)
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scm
from params import nt,t_period,t_final

def Vol(t,lake_vol_0):
    # define sinusoidal volume change
    return (0.45*lake_vol_0)*np.cos(2*np.pi*t/(t_period))

def Vdot(lake_vol_0,t):
    # compute rate of subglacial lake volume change
    dt_fine = 3.154e7/5000.0       # timestep for computing derivative (1/5000 yr)
    Vd = scm.derivative(Vol,t,dx=dt_fine,args=(lake_vol_0,))
    return Vd
