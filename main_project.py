from state_eqns import Afn
import scipy.integrate as integrate
import scipy
import numpy as np
import orbit_fns
from getDensityParams import getDensityParams
import math
import ray
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
import munkres as MR
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from scipy.stats.distributions import chi2
import copy
import pyproj
from sensor import Sensor
import json


ray.init()
rad2deg = 57.2958
deg2rad = 0.0174533
fov_params = [120*deg2rad,30*deg2rad]
np.random.seed(1)


def write_JSON(filename,data):
    with open(filename,'w') as outfile:
        json.dump(stringify_keys(data), outfile)

def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d

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

def save_Local_Belief(sensor_list):
	sensor_belief = dict()
	for s_i in sensor_list:
		sensor_belief.update({s_i.sensor_id:s_i.local_belief_history})
	write_JSON('Sensor_beliefs.json',sensor_belief)


debris_track = np.genfromtxt('project_truedata_long.dat',delimiter=',')
sensor_raw = np.genfromtxt('SensorSiteDetails_small.csv', delimiter=',')
sensor_sites_llh = dict([[int(ind_j),[j[5],j[6],j[7]]] for ind_j,j in enumerate(sensor_raw)])
sensor_sites_ecef = dict([[j,LLH2ECEF(sensor_sites_llh[j][0],sensor_sites_llh[j][1],sensor_sites_llh[j][2])] for j in sensor_sites_llh])
sensor_sites_z = dict([[j,measurement(sensor_sites_ecef[j])] for j in sensor_sites_ecef])
with open('sensors_observations_long.json') as json_file:
	sensor_observations = json.load(json_file)

num_pmf = 20
init_conditions = np.array([-2011.990,-382.065,6316.376,5.419783,-5.945319,1.37398]) ## km, km/s
P0 = np.block([[np.eye((3)),np.zeros((3,3))],[np.zeros((3,3)),1e-6*np.eye(3)]])
GMM = dict()
GMM_means = []
GMM_covars = np.zeros((6,6,num_pmf))
GMM_weights = 1.0/num_pmf*np.ones((1,num_pmf))
for j in range(num_pmf):
	GMM_covars[:,:,j] = P0
	GMM_means.append(np.random.multivariate_normal(init_conditions,P0))
GMM = dict([['means',np.array(GMM_means).T],['weights',GMM_weights],['covars',GMM_covars]])

t_f = 86401

interval = 5*60
time_range = list(range(0,t_f,interval))
sig_alpha = 10*4.848e-2
R = sig_alpha**2*np.eye(2)
Q = 1e-16*np.eye(3)
sensor_list = [Sensor(str(j),sensor_sites_llh[j],Q,R) for j in sensor_sites_llh]
# sensor_list = [sensor_list[0]]
for s_i in sensor_list:
	s_i.initialize(GMM,debris_track)
	s_i.initialize_data_output(time_range)

prev_t = 0
for t_i,t_k in enumerate(time_range):
	for s_i in sensor_list:
		z_k = sensor_observations[s_i.sensor_id].get(str(t_k))
		s_i.GMM_UKF(t_k,t_i,z_k=z_k)
	prev_t = t_k
	print(prev_t)
save_Local_Belief(sensor_list)

