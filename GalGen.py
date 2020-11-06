import os
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

#own imports
import constants as const



########################################
##    POSITION GENERATOR FUNCTIONS ##
########################################

# function to generate N random unit vectors
def randUnitVec(N):
	φ = np.random.uniform(low=0, high=2 * np.pi, size=N)
	cosθ = np.random.uniform(low=-1, high=1, size=N)
	θ = np.arccos(cosθ)
	x = np.sin(θ) * np.cos(φ)
	y = np.sin(θ) * np.sin(φ)
	z = np.cos(θ)
	uvec = np.stack((x, y, z), axis=1)
	return uvec

# function to generate N random particle positions according to the Plummer
# density profile
def gen3DPlummer(N, r0):
	f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
	r = r0 / np.sqrt(f ** (-2 / 3) - 1)
	return np.einsum('i,ij->ij', r, randUnitVec(N))

#function to generate N random particle positions according to the Jaffe
#density profile
def gen3DJaffe(N, r0):
	f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
	r = r0 * f / (1 - f)
	return np.einsum('i,ij->ij', r, randUnitVec(N))

#function to generate N random particle positions according to the Hernquist
#density profile
def gen3DHernquist(N, r0):
	f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
	r = r0 * (f + np.sqrt(f)) / (1 - f)
	return np.einsum('i,ij->ij', r, randUnitVec(N))

#function to generate N random particle positions according to a uniform disk
#density profile
def gen3DUniformDisk(N, r0):
	#NOTE: r0 = [rmin, rmax] where 
	f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
	r = np.sqrt(f * (r0[1] ** 2 - r0[0] ** 2) + r0[0] ** 2)
	φ = np.random.uniform(low=0, high=2 * np.pi, size=N)
	
	#generate a random unit vector in the x,y plane
	x = np.cos(φ)
	y = np.sin(φ)
	z = np.zeros(N)
	uvec = np.stack((x,y,z), axis=1)

	return np.einsum('i,ij->ij', r, uvec)

# handler function for generating N random particle positions
def generate_r(Npart, r0=None, type_='plummer'):
	if r0 is None:
		raise ValueError("r0 must be defined!")

	if type_ == "plummer":
		p = gen3DPlummer(Npart, r0)

	if type_ == "jaffe":
		p = gen3DJaffe(Npart, r0)

	if type_ == "hernquist":
		p = gen3DHernquist(Npart, r0)

	if type_ == "disk":
		p = gen3DUniformDisk(Npart, r0)

	return p
	

##############################
##    ROTATION FUNCTIONS    ##
##############################
#rotation around x axis
def Rx(θ):
	return np.array([[1, 0, 0],
					 [0, np.cos(θ), -np.sin(θ)],
					 [0, np.sin(θ), np.cos(θ)]])

#rotation around y axis
def Ry(θ):
	return np.array([[np.cos(θ), 0, np.sin(θ)],
					 [0, 1, 0],
					 [-np.sin(θ), 0, np.cos(θ)]])

#rotation around z axis
def Rz(θ):
	return np.array([[np.cos(θ), -np.sin(θ), 0],
					 [np.sin(θ), np.cos(θ), 0],
					 [0, 0, 1]])

#rotation operation
def rotate(pnts, θ, axis=None):
	if axis is None:
		raise ValueError("'axis' is a required argument!")
	if axis == 'x':
		return np.einsum('ij,kj->ki', Rx(θ), pnts)
	if axis == 'y':
		return np.einsum('ij,kj->ki', Ry(θ), pnts)
	if axis == 'z':
		return np.einsum('ij,kj->ki', Rz(θ), pnts)


########################################
##    VELOCITY GENERATOR FUNCTIONS ##
########################################

# Escape velocity function according to the Plummer model
def vesc_Plummer(r, M, r0):
	return np.sqrt(2 * const.G_ * M / r0) * (1 + r ** 2 / r0 ** 2) ** (-1 / 4)

# Escape velocity function according to the Jaffe model
def vesc_Jaffe(r, M, r0):
	return np.sqrt(2 * const.G_ * M / r0) * (-np.log(r / (r0 + r))) ** (1 / 2)

# Escape velocity function according to the Hernquist model
def vesc_Hernquist(r, M, r0):
	return np.sqrt(2 * const.G_ * M / r0) * (1 + r / r0) ** (-1 / 2)

# Function to generate non-radial velocity vectors
def vcirc(r, M, r0, ζ=1, type_="plummer"):
	if type_ == "disk":
		mag_v = 2 / np.pi * 30 * np.tanh(np.linalg.norm(r, axis=1) / ((r0[1] - r0[0]) / 2))
		if abs(ζ) == 1:
			vx = -1 * mag_v * ζ * r[:,1] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
			vy = mag_v * ζ * r[:,0] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
			vz = np.zeros(r.shape[0])
		else:
			ζ = np.random.choice([-1, 1], size=r.shape[0])
			vx = -1 * mag_v * ζ * r[:,1] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
			vy = mag_v * ζ * r[:,0] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
			vz = np.zeros(r.shape[0])
		return np.stack((vx, vy, vz), axis=1)

	R = np.linalg.norm(r, axis=1) #compute the magnitude of r
	φ = np.arctan2(r[:,1], r[:,0]) #compute the azimuth angle (φ) corresponding to each position vector
	θ = np.arccos(r[:,2] / R) #compute the polar angle (θ) corresponding to
									  #each position vector
	#θ = np.full(r.shape[0], np.pi / 2) #assumes all particles lie on a disk!

	transf = np.array([[-np.sin(θ) * np.sin(φ), np.cos(θ) * np.cos(φ)],
					   [np.sin(θ) * np.cos(φ), np.cos(θ) * np.sin(φ)],
					   [np.zeros(r.shape[0]), -np.sin(θ)]]) #transformation matrix from (phi, theta) coordinates to (x, y, z)
	
	#randomly generate a velocity vector (v) tangent to the spherical surface
	f = np.random.uniform(low=0.7, high=0.9, size=r.shape[0])

	if type_ == "plummer":
		v_e = vesc_Plummer(np.linalg.norm(r, axis=1), M, r0)
	elif type_ == "jaffe":
		v_e = vesc_Jaffe(np.linalg.norm(r, axis=1), M, r0)
	elif type_ == "hernquist":
		v_e = vesc_Hernquist(np.linalg.norm(r, axis=1), M, r0)

	mag_v = f * v_e
	χ = np.random.uniform(low=0, high=2 * np.pi, size=r.shape[0])
	
	"""
		ζ : 0 (or anything else) no fixed rotation direction around polar axis
		ζ : -1 clockwise rotation around polar axis (i.e. East-West rotation)
		ζ : +1 anti-clockwise rotation around polar axis (i.e. West-East rotation)

	"""
	if abs(ζ) == 1:
		vφ = ζ * np.abs(mag_v * np.cos(χ))
	else:
		vφ = mag_v * np.cos(χ)
	vθ = mag_v * np.sin(χ)
	v = np.stack((vφ, vθ), axis=1)

	#transform the velocity vector (v) tangent to the spherical surface to a
	#cartesian coordinate vector (v_prime)
	v_prime = np.einsum('ij,kji->ik', v, transf)

	return v_prime

def generate_v(r, r0=None, Mtot=None, ζ=1, type_="plummer"):
	if r0 is None:
		raise ValueError("'r0' must be defined!")
	if type_ == "disk":
		assert isinstance(r0, (np.ndarray, list, tuple)), f"To generate disks r0 should be list, ndarray or tuple, not {type(r0)}!"
	if Mtot is None:
		raise ValueError("'Mtot' must be defined!")

	v = vcirc(r, Mtot, r0, ζ = ζ, type_=type_)
	#v = vcirc_test(r, Mtot, r0, ζ = ζ, type_=type_)

	return v


#############################
##    GENERATOR HANDLER    ##
#############################
def generate(N, Mtot, r0, ζ=1, type_="plummer"):
	# N: number of particles to generate
	# Mtot: total mass of the Galaxy
	# r0: scaling radius of the Galaxy
	r = generate_r(N, r0=r0, type_=type_)
	v = generate_v(r, r0=r0, Mtot=Mtot, ζ = 1, type_=type_)
	return r, v


###################################################
##    FUNCTION FOR PLOTTING GENERATOR RESULTS ##
###################################################
def GeneratorPlot(p, window=None, type_="spatial", histograms=False, outpath=None):
	plt.style.use("dark_background")
	fig1 = plt.figure(figsize=(10,10))
	ax = fig1.add_subplot(111, projection='3d')
	ax.scatter(p[:,0], p[:,1], p[:,2], s=1, color='white')
	
	if type_ == "spatial":
		if window is not None:
			lim = window
		else:
			mean_p = np.mean(p, axis=0)
			max_std_p = np.max(np.std(p, axis=0))
			xlim, ylim, zlim = np.array([mean_p - 2*max_std_p, mean_p + 2*max_std_p]).T
		ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
		ax.set_xlabel(r"$x$ [kpc]", fontsize=16)
		ax.set_ylabel(r"$y$ [kpc]", fontsize=16)
		ax.set_zlabel(r"$z$ [kpc]", fontsize=16)
	elif type_ == "velocity":
		ax.set_xlabel(r"$v_x$ [kpc/Gyr]", fontsize=16)
		ax.set_ylabel(r"$v_y$ [kpc/Gyr]", fontsize=16)
		ax.set_zlabel(r"$v_z$ [kpc/Gyr]", fontsize=16)

	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False

	if histograms and type_ == "spatial":
		plt.style.use("default")
		fig2 = plt.figure(figsize=(10,10))

		histSetup = dict(bins=p.shape[0] // 100, density=True, rwidth=0.8) #color="white",
		
		#histogram for r
		r = np.linalg.norm(p, axis=1)
		mean_r = np.mean(r)
		std_r = np.std(r)
		rlim = (mean_r - 2*std_r, mean_r + 2*std_r)
		r_ = r[np.where(r > rlim[0])]
		r_ = r_[np.where(r_ < rlim[1])]
		ax1 = fig2.add_subplot(221)
		ax1.hist(r_, **histSetup)
		ax1.set_xlabel(r"$r$ [kpc]", fontsize=16)
		ax1.set_xlim(rlim)

		#histogram for phi
		phi = np.arctan2(p[:,1], p[:,0])
		philim = [-np.pi, np.pi]
		ax2 = fig2.add_subplot(222)
		ax2.hist(phi, **histSetup)
		ax2.set_xlabel(r"$\phi$ [rad]", fontsize=16)
		ax2.set_xlim(philim)

		#histogram for theta
		theta = np.arctan2(np.linalg.norm(p[:,:-1], axis=1), p[:,2])
		thetalim = [0, np.pi]
		ax3 = fig2.add_subplot(223)
		ax3.hist(theta, **histSetup)
		ax3.set_xlabel(r"$\theta$ [rad]", fontsize=16)
		ax3.set_xlim(thetalim)

	if outpath is not None:
		fig1.savefig(os.path.join(outpath, f"{type_}_scatter.pdf"), dpi=fig1.dpi, bbox_inches=fig1.get_tightbbox(fig1.canvas.get_renderer()))
		if type_ == "spatial":
			fig2.savefig(os.path.join(outpath, f"{type_}_histogram.pdf"), dpi=fig2.dpi, bbox_inches=fig2.get_tightbbox(fig2.canvas.get_renderer()))
	plt.show()

if __name__ == "__main__":
	Nparticles = 10000
	Mtot = 10 ** 8
	r0 = [2, 15] #20

	θ = (np.pi / 4 , np.pi / 4, np.pi / 4)

	r, v = generate(Nparticles, Mtot, r0, type_="disk")
	
	#rotate r and v
	r = rotate(rotate(rotate(r, θ[0], axis='x'), θ[1], axis='y'), θ[2], axis='z')
	v = rotate(rotate(rotate(v, θ[0], axis='x'), θ[1], axis='y'), θ[2], axis='z')
	
	GeneratorPlot(r, type_="spatial", histograms=True)
	GeneratorPlot(v, type_="velocity")