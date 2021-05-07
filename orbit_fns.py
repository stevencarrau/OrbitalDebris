import math
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm
from scipy.stats.distributions import chi2
mu = 398600.4415
Rearth = 6378.1363

def cart2kep(r,v):
	## Returns a, e, i, Omega & omega
	rad = np.linalg.norm(r)
	vel = np.linalg.norm(v)
	a = rad/(2-rad*vel**2/mu)
	mue = 1/mu*((vel**2-mu/rad)*r-np.dot(r,v)*v)
	e = np.linalg.norm(mue)
	h = np.cross(r,v)
	n = np.cross(np.array([0,0,1]).T,h)
	i = np.arccos(h[2]/np.linalg.norm(h))
	Omega = np.arccos(n[0]/np.linalg.norm(n)) if n[1]>=0 else 2*math.pi-np.arccos(n[0]/np.linalg.norm(n))
	omega = np.arccos(np.dot(n,mue)/(np.linalg.norm(n)*e)) if mue[2]>=0 else 2*math.pi - np.arccos(np.dot(n,mue)/(np.linalg.norm(n)*e))
	return a,e,i,Omega,omega

def radial_frame(relate_state,state_in,P_in):
	if state_in is type(list):
		r = np.array(relate_state[0:3]).reshape((3,1))
		v = np.array(relate_state[3:]).reshape((3,1))
	else:
		if len(np.shape(state_in)) > 1:
			for j in range(state_in.shape[0]):
				r = np.array(relate_state[j,0:3]).reshape((3,1))
				v = np.array(state_in[j,3:]).reshape((3,1))
		else:
			r = np.array(relate_state[0:3]).reshape((3,1))
			v = np.array(relate_state[3:]).reshape((3,1))

	r_hat = r/np.linalg.norm(r)
	c = np.cross(r.reshape(-1),v.reshape(-1))
	c_hat = c/np.linalg.norm(c)
	I_hat = np.cross(c_hat.reshape(-1),r_hat.reshape(-1))
	T_in = np.hstack((r_hat.reshape(3,1),I_hat.reshape(3,1))).T

	r_I = np.matmul(T_in,state_in[0:3])
	P_rI = np.matmul(T_in,np.matmul(P_in[0:3,0:3],T_in.T))
	return r_I,P_rI

def prob_ellipse(P):
	w,v = np.linalg.eig(P)
	return np.sqrt(w),np.arctan2(v[1,0],v[0,0])

def plot_ellipse(x,y,ax,n_std=3,color='none',P=np.eye(2)):
	conf = 2 * norm.cdf(n_std) - 1.0
	scale = chi2.ppf(conf, df=2)
	wh,angle = prob_ellipse(P*scale)
	ellipse = Ellipse((0,0),width=2*wh[0],height=2*wh[1],facecolor='none',edgecolor=color,linewidth=3)
	transf = transforms.Affine2D().rotate(angle).translate(x,y)
	ellipse.set_transform(transf+ax.transData)
	return ax.add_patch(ellipse)

def plot_ellipse_scale(x,y,ax,scale=3,color='none',P=np.eye(2)):
	wh,angle = prob_ellipse(P*scale)
	ellipse = Ellipse((0,0),width=2*wh[0],height=2*wh[1],facecolor='none',edgecolor=color,linewidth=1,linestyle='--')
	transf = transforms.Affine2D().rotate(angle).translate(x,y)
	ellipse.set_transform(transf+ax.transData)
	return ax.add_patch(ellipse)
