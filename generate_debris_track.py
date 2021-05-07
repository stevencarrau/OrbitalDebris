from state_eqns import Afn
import scipy.integrate as integrate
import numpy as np
from getDensityParams import getDensityParams
import math
import ray
from matplotlib import rc
import copy

data = np.genfromtxt('hw03_data.dat')
rc('font', **{'family':'serif','serif':['Palatino'],'size':16})
rc('text', usetex=True)
ray.init()
init_conditions = np.array([-2011.990,-382.065,6316.376,5.419783,-5.945319,1.37398]) ## km, km/s
P0 = np.block([[np.eye((3)),np.zeros((3,3))],[np.zeros((3,3)),1e-6*np.eye(3)]])
J2 = 0.0010826267
J3 = -0.0000025327
mu = 398600.4415
Rearth = 6378.1363
Cd = 2.0
A = 3.6e-6
m = 1350
thetadot = 7.29211585530066e-5
tf = 86400
Jmax = 800
sig_alpha = 10*4.848e-2
R = sig_alpha**2*np.eye(2)
Q = 1e-16*np.eye(3)


def deriv_noise(t_in,r_in,v_in):
	r = r_in[0:3]
	rdot = r_in[3:6]
	h = np.linalg.norm(r) - Rearth
	rho0, h0, H = getDensityParams(h)
	rhoA = rho0*math.exp(-(h-h0)/H)*1e9
	drag_co = -0.5*Cd*A/m*rhoA
	rddot_eval,Aeval = Afn(r[0],r[1],r[2],rdot[0],rdot[1],rdot[2],drag_co)
	dR = np.array([rdot[0],rdot[1],rdot[2],rddot_eval[0]+v_in[0],rddot_eval[1]+v_in[1],rddot_eval[2]+v_in[2]])
	return dR

def measurement(x_in):
	r = np.linalg.norm(x_in[0:3])
	delta = math.asin(x_in[2] / r)
	alpha = math.acos(x_in[0]/(r*math.cos(delta))) if x_in[1]>=0 else -math.acos(x_in[0]/(r*math.cos(delta)))
	return np.array([alpha,delta]).T

def propagate(ICs,tf):
	## Time update
	if len(ICs) <9:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0, tf], y0=ICs[0:6], args=([np.zeros((3,))]), method='DOP853',
									rtol=1e-10, atol=1e-12)
	else:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0,tf], y0=ICs[0:6],args=([ICs[6:9]]),method='DOP853',rtol=1e-10,atol=1e-12)
	xout = r_out.y.T[-1][:6] if tf!=0 else ICs[0:6]
	return xout

true_x0 = np.zeros((7,))
## Start at true data value from HW3
true_x0[1:] = data[0,3:]
debris_track = [copy.deepcopy(true_x0)]
interval = 5*60
for t_i in range(interval,tf+1,interval):
	true_x0[1:] = propagate(true_x0[1:],interval)
	true_x0[0] = t_i
	debris_track.append(copy.deepcopy(true_x0))

np.savetxt('project_truedata_long.dat',debris_track,delimiter=',')