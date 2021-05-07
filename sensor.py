import numpy as np
import math
import pyproj
import ray
import scipy.integrate as integrate
from state_eqns import Afn
from getDensityParams import getDensityParams
import scipy
import copy
import itertools



J2 = 0.0010826267
J3 = -0.0000025327
mu = 398600.4415
Rearth = 6378.1363
Cd = 2.0
A = 3.6e-6
m = 1350
thetadot = 7.29211585530066e-5
tf = 86400
sig_alpha = 10*4.848e-2
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

rad2deg = 57.2958
deg2rad = 0.0174533
fov_params = [120*deg2rad,30*deg2rad]


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

def update(ICs,tf,prev_t):
	ti = tf-prev_t
	## Time update
	if len(ICs) <9:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0, ti], y0=ICs[0:6], args=([np.zeros((3,))]), method='DOP853',
									rtol=1e-10, atol=1e-12)
	else:
		r_out = integrate.solve_ivp(deriv_noise, t_span=[0,ti], y0=ICs[0:6],args=([ICs[6:9]]),method='DOP853',rtol=1e-10,atol=1e-12)
		# k1+=30
		# ICs[0:6] = r_out.y.T[-1,:6]
	# output_res[i, 0] = r_out.t[-1]
	xout = r_out.y.T[-1][:6] if tf!=0 else ICs[0:6]
	## Measurement update
	z_sig = measurement(ECEF2ECI(xout[0:3],tf)) + ICs[-2:]
	return xout,z_sig



def rotation3(angle):
	mat = np.zeros((3,3))
	mat[2,2] = 1
	mat[0,0] = math.cos(angle)
	mat[1,1] = math.cos(angle)
	mat[0,1] = math.sin(angle)
	mat[1,0] = -math.sin(angle)
	return mat

def ECEF2ECI(ecef,t):
	omega_ie = 7.292115e-5 #rad/s
	rot_angle = omega_ie*t
	return np.matmul(np.linalg.inv(rotation3(rot_angle)),ecef)

def ECI2ECEF(eci,t):
	omega_ie = 7.292115e-5  # rad/s
	rot_angle = omega_ie * t
	return np.matmul(rotation3(rot_angle), eci)

def LLH2ECEF(lat,lon,h):
	ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
	lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
	ecef_vec = np.zeros((3,))
	t = pyproj.Transformer.from_proj(lla, ecef)
	x,y,z = t.transform(lon,lat,h,radians=False)
	ecef_vec[0] = x/1e3
	ecef_vec[1] = y/1e3
	ecef_vec[2] = z/1e3
	return ecef_vec

def ECEF2LLH(X,Y,Z):
	ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
	lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
	t = pyproj.Transformer.from_proj(ecef, lla)
	lla_vec = np.zeros((3,))
	lon,lat,alt = t.transform(X*1e3,Y*1e3,Z*1e3,radians=False)
	lla_vec[0] = lat
	lla_vec[1] = lon
	lla_vec[2] = alt
	return lla_vec

def measurement(x_in):
	r = np.linalg.norm(x_in[0:3])
	delta = math.asin(x_in[2] / r)
	alpha = math.acos(x_in[0]/(r*math.cos(delta))) if x_in[1]>=0 else -math.acos(x_in[0]/(r*math.cos(delta)))
	return np.array([alpha,delta]).T

@ray.remote
def UKF(sigma_points,z_sigma, t_k,prev_t,z_k=None):
	new_sigma = copy.deepcopy(sigma_points)
	new_z_sigma = copy.deepcopy(z_sigma)
	## Time Update
	for c_s in range(sigma_points.shape[0]):
		# print('{}  {}  {}'.format(t_k,prev_t,sigma_points.shape[0]))
		xout, zout = update(sigma_points[c_s, :], t_k,prev_t)
		new_sigma[c_s, :6] = xout
		new_z_sigma[c_s, :] = zout
	sigma_points = new_sigma
	z_sigma = new_z_sigma
	xbar = np.matmul(sigma_weights, sigma_points[:, :6])
	Pbar = np.zeros((6,6))
	for i in range(sigma_points.shape[0]):
		Pbar += cov_weights[i] * np.matmul(np.reshape(sigma_points[i, :6] - xbar, (6, 1)),np.reshape(sigma_points[i, :6] - xbar, (1, 6)))
	if z_k:
		## Measure update
		zbar = np.matmul(sigma_weights, z_sigma)
		Pzz = np.zeros_like(R)
		Pxz = np.zeros((Pbar.shape[0], Pzz.shape[1]))
		for i in range(sigma_points.shape[0]):
			Pzz += cov_weights[i] * np.matmul(np.reshape(z_sigma[i, :] - zbar, (2, 1)),np.reshape(z_sigma[i, :] - zbar, (1, 2)))
			Pxz += cov_weights[i] * np.matmul(np.reshape(sigma_points[i, :6] - xbar, (6, 1)),np.reshape(z_sigma[i, :] - zbar, (1, 2)))
		K = np.matmul(Pxz, np.linalg.inv(Pzz))
		e_bar = z_k - zbar
		x_hat = xbar + np.matmul(K, e_bar)
		Pk = Pbar - np.matmul(K, np.matmul(Pzz, K.T))
		return sigma_points,z_sigma,x_hat,Pk,zbar,Pzz
	else:
		return sigma_points,z_sigma

class Sensor():
	def __init__(self,sensor_id,llh,Q,R):
		self.sensor_id = sensor_id
		self.llh = llh
		self.ecef = LLH2ECEF(llh[0],llh[1],llh[2])
		self.z_s = measurement(self.ecef)
		self.Q = Q
		self.R = R
		self.prev_t = 0

	def initialize(self,GMM,true_data):
		self.number_bins = len(GMM['weights'])
		self.pmf = GMM['weights']
		self.GMM_means = GMM['means']
		self.GMM_covars = GMM['covars']
		self.GMM_estimate()
		self.sigma_points = []
		self.z_sigmas = []
		for i in range(self.GMM_means.shape[1]):
			sigma_points,z_sigma = self.preprocess_sigma_points(self.GMM_means[:,i],self.GMM_covars[:,:,i])
			self.sigma_points.append(sigma_points)
			self.z_sigmas.append(z_sigma)
		self.true_data = true_data
		self.local_belief_history = dict()
		self.track_history = dict()

	def preprocess_sigma_points(self,xbar,Pbar):
		Phat = scipy.linalg.block_diag(Pbar, self.Q, self.R)
		L_tri = np.linalg.cholesky(Phat)
		xbar_ext = np.concatenate((xbar, np.zeros((3,)), np.zeros((2,))))
		sigma_points = np.array([xbar_ext] + list(itertools.chain.from_iterable(
			[[xbar_ext + math.sqrt(nprime + lam) * L_tri[:, i]] + [xbar_ext - math.sqrt(nprime + lam) * L_tri[:, i]]
			 for i in range(L_tri.shape[1])])))
		z_sigma = np.zeros((sigma_points.shape[0], 2))
		return sigma_points,z_sigma

	def initialize_data_output(self,time_range):
		self.output_data = np.zeros((len(time_range), 1 + 6 + 6))
		self.output_data[:,0] = time_range

	def GMM_estimate(self):
		self.x_bar = np.add.reduce([self.pmf[:,i] * self.GMM_means[:,i] for i in range(self.pmf.shape[1])]).reshape((-1, 1))
		self.P_k = np.add.reduce([self.pmf[:,i] * (self.GMM_covars[:,:,i] + np.matmul(self.GMM_means[:,i].reshape((-1, 1)) - self.x_bar, (self.GMM_means[:,i].reshape((-1, 1)) - self.x_bar).T)) for i in range(self.pmf.shape[1])])



	def GMM_UKF(self,t_k,t_i,z_k=None):
		output_traj = []
		if z_k:
			for i in range(self.GMM_means.shape[1]):
				output_traj.append(UKF.remote(self.sigma_points[i], self.z_sigmas[i],t_k,self.prev_t, z_k))
			output_list = ray.get(output_traj)
			s_points = []
			z_points = []
			wtilde = np.zeros_like(self.pmf)
			x_set = np.zeros_like(self.GMM_means)
			P_set = np.zeros_like(self.GMM_covars)
			for i,out_line in enumerate(output_list):
				sigma_points,z_sigma,x_hat,Pk,zbar,Pzz = out_line
				s_points.append(sigma_points)
				z_points.append(z_sigma)
				wtilde[:,i] =self.pmf[:,i]*scipy.stats.multivariate_normal.pdf(z_k, mean=zbar, cov=Pzz)
				x_set[:,i] = x_hat
				P_set[:,:,i] = Pk

			wsum = np.sum(wtilde)
			self.pmf = wtilde/wsum
			self.GMM_means = x_set
			self.GMM_covars = P_set
			self.GMM_estimate()

		else:
			for i in range(self.GMM_means.shape[1]):
				output_traj.append(UKF.remote(self.sigma_points[i], self.z_sigmas[i],t_k,self.prev_t, z_k))
			output_list = ray.get(output_traj)
			s_points = []
			z_points = []
			for weight, out_line in zip(np.array(self.pmf).flatten(), output_list):
				sigma_points,z_sigma = out_line
				s_points.append(sigma_points)
				z_points.append(z_sigma)
			self.sigma_points = s_points
			self.z_sigmas = z_points
			self.GMM_estimate()
		self.output_data[t_i, 1:7] = np.array(self.true_data[t_i,1:]) - self.x_bar.flatten()
		self.output_data[t_i, 7:13] = np.sqrt(np.diag(self.P_k))
		self.local_belief_history.update({t_k:self.pmf.tolist()})
		self.track_history.update({t_k:self.GMM_means.tolist()})
		self.prev_t = t_k
