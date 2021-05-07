from state_eqns import Afn
import scipy.integrate as integrate
import numpy as np
from getDensityParams import getDensityParams
import math
import ray
from matplotlib import rc
import copy
import matplotlib.pyplot as plt
import pyproj
import json


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


rc('font', **{'family':'serif','serif':['Palatino'],'size':16})
rc('text', usetex=True)
debris_track = np.genfromtxt('project_truedata_long.dat',delimiter=',')
sensor_raw = np.genfromtxt('SensorSiteDetails_small.csv', delimiter=',')
sensor_sites_llh = dict([[int(ind_j),[j[5],j[6],j[7]]] for ind_j,j in enumerate(sensor_raw)])
rad2deg = 57.2958
deg2rad = 0.0174533
fov_params = [120*deg2rad,30*deg2rad]
sig_alpha = 10*4.848e-6
R_noise = sig_alpha**2*np.eye(2)

#
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


sensor_sites_ecef = dict([[j,LLH2ECEF(sensor_sites_llh[j][0],sensor_sites_llh[j][1],sensor_sites_llh[j][2])] for j in sensor_sites_llh])


def measurement(x_in):
	r = np.linalg.norm(x_in[0:3])
	delta = math.asin(x_in[2] / r)
	alpha = math.acos(x_in[0]/(r*math.cos(delta))) if x_in[1]>=0 else -math.acos(x_in[0]/(r*math.cos(delta)))
	return np.array([alpha,delta]).T

sensor_sites_z = dict([[j,measurement(sensor_sites_ecef[j])] for j in sensor_sites_ecef])

def in_range(z,z_sen,fov_params):
	alpha_p = (math.pi+z_sen[0] + fov_params[0]/2) % (2*math.pi) - math.pi
	alpha_m = (math.pi+z_sen[0] - fov_params[0]/2) % (2*math.pi) - math.pi
	if alpha_p < alpha_m:
		if min(alpha_m,math.pi) < z[0] < max(alpha_m,math.pi) or min(alpha_p,-math.pi) < z[0] < max(alpha_p,-math.pi):
			alpha_flag = True
		else:
			return False
	else:
		if min(alpha_m,alpha_p) < z[0] < max(alpha_m,alpha_p):
			alpha_flag = True
		else:
			return False

	delta_p = (math.pi+z_sen[1] + fov_params[1]/2) % (2*math.pi) - math.pi
	delta_m = (math.pi+z_sen[1] - fov_params[1]/2) % (2*math.pi) - math.pi
	if delta_p < delta_m:
		if min(delta_m,math.pi) < z[1] < max(delta_m,math.pi) or min(delta_p,-math.pi) < z[1] < max(delta_p,-math.pi):
			return True
		else:
			return False
	else:
		if min(delta_m, delta_p) < z[1] < max(delta_m, delta_p):
			return True
		else:
			return False

measure_array = np.array([measurement(x_in[1:]) for x_in in debris_track])
eci_track = np.array([ECEF2ECI(x_in[1:4],x_in[0]) for x_in in debris_track])
# llh_track = np.array([ECEF2LLH(x_in[0],x_in[1],x_in[2]) for x_in in eci_track])
# fig,ax = plt.subplots(1,1,figsize=(7.5,5.65))
# plt.imshow(plt.imread('earthmap.png'),extent=[-180,180,-90,90])
# ax.plot(llh_track[:,1],llh_track[:,0],linestyle='None',marker='o',markeredgecolor='r',markerfacecolor='r',markersize=1)
# ax.plot(llh_track[0,1],llh_track[0,0],linestyle='None',marker='o',markeredgecolor='yellow',markerfacecolor='yellow',markersize=4)
# ax.plot(llh_track[-1,1],llh_track[-1,0],linestyle='None',marker='o',markeredgecolor='k',markerfacecolor='k',markersize=4)
# ax.plot(np.array(list(sensor_sites_llh.values()))[:,1],np.array(list(sensor_sites_llh.values()))[:,0],linestyle='None',marker='o',markeredgecolor='magenta',markerfacecolor='magenta',markersize=4)
# ax.set_xlabel(r'Longitude ($\lambda^{\circ}$)')
# ax.set_ylabel(r'Latitude ($\phi^{\circ}$)')
# # ax.plot(debris_track[:,0],measure_array[:,0],linestyle='None',marker='o',markeredgecolor='b',markerfacecolor='b',markersize=1)
# # ax.plot(debris_track[:,0],measure_array[:,1],linestyle='None',marker='o',markeredgecolor='r',markerfacecolor='r',markersize=1)
# plt.savefig('Satellite_Track_long.pdf')

observations = dict([[i,dict()] for i in sensor_sites_z])
for sen in sensor_sites_z:
	z_s = sensor_sites_z[sen]
	repeats = False
	observations[sen].update({0:list(measurement(debris_track[0][1:]))})
	for ind_j,i in enumerate(debris_track):
		z_rel = measurement(ECEF2ECI(i[1:4],i[0]))
		z_i = measurement(i[1:4])
		if in_range(z_rel,z_s,fov_params):
			if not repeats:
				repeats = True
				new_measure = np.random.multivariate_normal(z_i,R_noise)
				# new_measure[0] = (math.pi+new_measure[0]) % (2*math.pi) - math.pi
				# new_measure[1] = (math.pi+new_measure[1]) % (2*math.pi) - math.pi
				measure = {int(i[0]):list(new_measure)}
				observations[sen].update(measure)
		else:
			repeats = False

write_JSON('sensors_observations_long.json',observations)