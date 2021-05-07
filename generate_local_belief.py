from state_eqns import Afn
import scipy.integrate as integrate
import scipy
import numpy as np
from getDensityParams import getDensityParams
import math
import ray
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.io import loadmat
import json
from scipy.stats.distributions import chi2
import pickle



GMM = loadmat('hw03_gmm.mat')
data = np.genfromtxt('hw03_data.dat')

num_pmf = 7
init_conditions = np.array(data[0,3:]) ## km, km/s
offset_initial = np.array([-2011.990,-382.065,6316.376,5.419783,-5.945319,1.37398])
P0 = np.block([[50*np.eye((3)),np.zeros((3,3))],[np.zeros((3,3)),50e-6*np.eye(3)]])
GMM = dict()
GMM_means = []
GMM_covars = np.zeros((6,6,num_pmf+1))
GMM_weights = 1.0/(num_pmf+1)*np.ones((1,num_pmf+1))
for j in range(num_pmf):
	GMM_covars[:,:,j] = P0
	GMM_means.append(np.random.multivariate_normal(offset_initial,P0))
GMM_covars[:,:,-1] = P0
GMM_means.append(init_conditions)

GMM = dict([['means',np.array(GMM_means).T],['weights',GMM_weights],['covars',GMM_covars]])

rc('font', **{'family':'serif','serif':['Palatino'],'size':16})
rc('text', usetex=True)
ray.init()
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
sig_alpha = 10*4.848e-6
R = sig_alpha**2*np.eye(2)
Q = 1e-16*np.eye(3)

## UT
beta = 2
alpha = 1
n = 6
nprime = n+Q.shape[0]+R.shape[0]
kappa = 3-nprime
lam = alpha**2*(nprime+kappa)-nprime
sigma_weights = np.zeros((2*nprime+1,))
sigma_weights[0] = lam/(nprime+lam)
sigma_weights[1:] = 1/(2*(nprime+lam))
cov_weights = np.zeros((2*nprime+1,))
cov_weights[0] = lam/(nprime+lam)+(1-alpha**2+beta)
cov_weights[1:] = 1/(2*(nprime+lam))

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

def update(ICs,tf):
	## Time update
	if len(ICs) <9:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0, tf], y0=ICs[0:6], args=([np.zeros((3,))]), method='DOP853',
									rtol=1e-10, atol=1e-12)
	else:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0,tf], y0=ICs[0:6],args=([ICs[6:9]]),method='DOP853',rtol=1e-10,atol=1e-12)
		# k1+=30
		# ICs[0:6] = r_out.y.T[-1,:6]
	# output_res[i, 0] = r_out.t[-1]
	xout = r_out.y.T[-1][:6] if tf!=0 else ICs[0:6]
	## Measurement update
	z_sig = measurement(xout) + ICs[-2:]
	return xout,z_sig

def measurement(x_in):
	r = np.linalg.norm(x_in[0:3])
	delta = math.asin(x_in[2] / r)
	alpha = math.acos(x_in[0]/(r*math.cos(delta))) if x_in[1]>=0 else -math.acos(x_in[0]/(r*math.cos(delta)))
	return np.array([alpha,delta]).T

@ray.remote
def UKF(x_hat,Pk,z_k,k,prev_t=0):
	Phat = scipy.linalg.block_diag(Pk, Q, R)
	L_tri = np.linalg.cholesky(Phat)
	xbar_ext = np.concatenate((x_hat, np.zeros((3,)), np.zeros((2,))))
	sigma_points = np.array([xbar_ext] + list(itertools.chain.from_iterable(
		[[xbar_ext + math.sqrt(nprime + lam) * L_tri[:, i]] + [xbar_ext - math.sqrt(nprime + lam) * L_tri[:, i]] for i
		 in
		 range(L_tri.shape[1])])))
	z_sigma = np.zeros((sigma_points.shape[0], 2))
	for c_s in range(sigma_points.shape[0]):
		xout,zout = update(sigma_points[c_s,:],k-prev_t)
		sigma_points[c_s,:6] = xout
		z_sigma[c_s,:] = zout
	xbar = np.matmul(sigma_weights,sigma_points[:,:6])
	zbar = np.matmul(sigma_weights,z_sigma)
	Pbar = np.zeros_like(P0)
	Pzz = np.zeros_like(R)
	Pxz = np.zeros((Pbar.shape[0],Pzz.shape[1]))
	for i in range(sigma_points.shape[0]):
		Pbar += cov_weights[i]*np.matmul(np.reshape(sigma_points[i,:6]-xbar,(6,1)),np.reshape(sigma_points[i,:6]-xbar,(1,6)))
		Pzz += cov_weights[i]*np.matmul(np.reshape(z_sigma[i,:]-zbar,(2,1)),np.reshape(z_sigma[i,:]-zbar,(1,2)))
		Pxz += cov_weights[i]*np.matmul(np.reshape(sigma_points[i,:6]-xbar,(6,1)),np.reshape(z_sigma[i,:]-zbar,(1,2)))
	K = np.matmul(Pxz,np.linalg.inv(Pzz))
	e_bar = z_k - zbar
	x_hat = xbar + np.matmul(K,e_bar)
	Pk = Pbar - np.matmul(K,np.matmul(Pzz,K.T))
	return x_hat,Pk,zbar,Pzz


def GMM_UKF(measurement_data,GMM_weights,GMM_means,GMM_covars):
	prev_t = 0
	output_data = dict()
	# output_data = np.zeros((measurement_data.shape[0],1+6+6))

	for k in range(0, measurement_data.shape[0]):
		x_set = []
		P_set = []
		I_set = set(range(len(GMM_weights)))
		wtilde = []
		output_traj = []
		for weight, mean, cov in zip(GMM_weights, GMM_means, GMM_covars):
			output_traj.append(UKF.remote(mean, cov,measurement_data[k, 1:3],measurement_data[k, 0], prev_t))
		output_list = ray.get(output_traj)

		for weight, out_line in zip(np.array(GMM_weights).flatten(), output_list):
			x_hat, Pk, zbar, Pzz = out_line
			x_set.append(x_hat)
			P_set.append(Pk)
			# pdf_k += weight*scipy.stats.multivariate_normal.pdf(mean=x_hat,cov=Pk)
			wtilde.append(weight * scipy.stats.multivariate_normal.pdf(measurement_data[k, 1:3], mean=zbar, cov=Pzz))
		wsum = sum(wtilde)
		print(wsum)
		GMM_weights = [w_t / wsum for w_t in wtilde]
		x_bar = np.add.reduce([w_s*x_h for w_s,x_h in zip(GMM_weights,x_set)]).reshape((-1,1))
		P_k = np.add.reduce([w_s*(P_s+np.matmul(x_s.reshape((-1,1))-x_bar,(x_s.reshape((-1,1))-x_bar).T)) for w_s,x_s,P_s in zip(GMM_weights,x_set,P_set)])
		GMM_means = [x_set[k_s] for k_s in I_set]
		GMM_covars = [P_set[k_s] for k_s in I_set]
		out_dict = dict([['xbar',x_bar],['Pk',P_k],['weights',GMM_weights],['means',GMM_weights],['covars',GMM_covars],['residuals',np.array(measurement_data[k,3:])-x_bar.flatten()]])
		output_data.update({int(measurement_data[k,0]):out_dict})
		prev_t = measurement_data[k, 0]
		print(prev_t)
	return output_data

debris_track = np.genfromtxt('project_truedata_long.dat',delimiter=',')
sensor_raw = np.genfromtxt('SensorSiteDetails_small.csv', delimiter=',')
sensor_sites_llh = dict([[int(ind_j),[j[5],j[6],j[7]]] for ind_j,j in enumerate(sensor_raw)])
with open('sensors_observations_fixed.json') as json_file:
	sensor_observations = json.load(json_file)


num_pmf = 7
init_conditions = debris_track[0,1:] ## km, km/s
P0 = np.block([[50*np.eye((3)),np.zeros((3,3))],[np.zeros((3,3)),50e-6*np.eye(3)]])
GMM = dict()
GMM_means = []
GMM_covars = np.zeros((6,6,num_pmf+1))
GMM_weights = 1.0/(num_pmf+1)*np.ones((1,num_pmf+1))
for j in range(num_pmf):
	GMM_covars[:,:,j] = P0
	GMM_means.append(np.random.multivariate_normal(init_conditions,P0))
GMM_covars[:,:,-1] = P0
GMM_means.append(init_conditions)
GMM = dict([['means',np.array(GMM_means).T],['weights',GMM_weights],['covars',GMM_covars]])

t_f = 86401
GMM_covars = [GMM['covars'][:,:,i] for i in range(GMM['covars'].shape[2])]
GMM_weights = [GMM['weights'][:,i] for i in range(GMM['weights'].shape[1])]
GMM_means = [GMM['means'][:,i] for i in range(GMM['means'].shape[1])]

sensor_measures = dict()
for s_i in sensor_sites_llh:
	measure_list = []
	for s_o in sensor_observations[str(s_i)]:
		time_s = np.argwhere(debris_track[:,0]==int(s_o))[0][0]
		time_line = [int(s_o)]+ sensor_observations[str(s_i)][s_o] + debris_track[time_s,1:].tolist()
		measure_list.append(time_line)
	sensor_measures.update({s_i:np.array(measure_list)})

local_belief_updates = dict()
for ind_s,s_i in enumerate(sensor_measures):
	output = GMM_UKF(sensor_measures[s_i],GMM_weights,GMM_means,GMM_covars)
	local_belief_updates.update({str(ind_s):output})

fileh = open('LocalBel.pkl','wb')
pickle.dump(local_belief_updates,fileh)
fileh.close()



# RMS_values_3 = [np.sqrt(np.mean(np.square(output_3[:,i_s]))) for i_s in range(1,7)]
# print(RMS_values_3)