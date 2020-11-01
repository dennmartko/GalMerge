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
    #NOTE: r0 is here the maximum radius of the disk!
    f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
    r = r0 * np.sqrt(f)
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
        mag_v = 120 * np.tanh(np.linalg.norm(r, axis=1) / r0)
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
    #θ = np.arccos(r[:,2] / R) #compute the polar angle (θ) corresponding to
                                      #each position vector
    θ = np.full(r.shape[0], np.pi / 2) #assumes all particles lie on a disk!

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

#TEST VERSION VCIRC
def vcirc_test(r, M, r0, ζ=1, type_="plummer"):
    #randomly generate a velocity vector (v) tangent to the spherical surface
    f = np.random.uniform(low=0.7, high=0.9, size=r.shape[0])

    if type_ == "plummer":
        v_e = vesc_Plummer(np.linalg.norm(r, axis=1), M, r0)
    elif type_ == "jaffe":
        v_e = vesc_Jaffe(np.linalg.norm(r, axis=1), M, r0)
    elif type_ == "hernquist":
        v_e = vesc_Hernquist(np.linalg.norm(r, axis=1), M, r0)
    elif type_ == "disk":
        v_e = np.zeros(r.shape[0])

    mag_v = f * v_e
        
    """
        ζ : 0 (or anything else) no fixed rotation direction around polar axis
        ζ : -1 clockwise rotation around polar axis (i.e. East-West rotation)
        ζ : +1 anti-clockwise rotation around polar axis (i.e. West-East rotation)

    """
    if abs(ζ) == 1:
        vx = -1 * mag_v * ζ * r[:,1] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
        vy = mag_v * ζ * r[:,0] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
        vz = np.zeros(r.shape[0])
    else:
        ζ = np.random.choice([-1, 1], size=r.shape[0])
        vx = -1 * mag_v * ζ * r[:,1] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
        vy = mag_v * ζ * r[:,0] / (r[:,0] ** 2 + r[:,1] ** 2) ** 0.5
        vz = np.zeros(r.shape[0])

    v = np.stack((vx, vy, vz), axis=1)

    return v


def generate_v(r, r0=None, Mtot=None, ζ=1, type_="plummer"):
    if r0 is None:
        raise ValueError("'r0' must be defined!")
    if Mtot is None:
        raise ValueError("'Mtot' must be defined!")

    v = vcirc(r, Mtot, r0, ζ = ζ, type_=type_)
    #v = vcirc_test(r, Mtot, r0, ζ = ζ, type_=type_)

    return v

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
def GeneratorPlot(p, type_="spatial", histograms=False):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2], s=1, color='white')
    
    if type_ == "spatial":
        ax.set(xlabel=r"$x$ (kpc)", ylabel=r"$y$ (kpc)", zlabel=r"$z$ (kpc)")
    elif type_ == "velocity":
        ax.set(xlabel=r"$v_x$ (kpc/Gyr)", ylabel=r"$v_y$ (kpc/Gyr)", zlabel=r"$v_z$ (kpc/Gyr)")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if histograms and type_ == "spatial":
        fig = plt.figure(figsize=(10,10))

        histSetup = dict(bins=p.shape[0] // 100, color="white", density=True)
        
        #histogram for r
        ax1 = fig.add_subplot(221)
        ax1.hist(np.linalg.norm(p, axis=1), **histSetup)
        ax1.set(xlabel=r"$r$ (kpc)")

        #histogram for phi
        ax2 = fig.add_subplot(222)
        ax2.hist(np.arctan2(p[:,1], p[:,0]), **histSetup)
        ax2.set(xlabel=r"$\phi$ (rad)")

        #histogram for theta
        ax3 = fig.add_subplot(223)
        ax3.hist(np.arctan2(np.linalg.norm(p[:,:-1], axis=1), p[:,2]), **histSetup)
        ax3.set(xlabel=r"$\theta$ (rad)")

    plt.show()

if __name__ == "__main__":
    Nparticles = 10000
    Mtot = 10 ** 8
    r0 = 10

    r, v = generate(Nparticles, Mtot, r0, type_="disk")
    GeneratorPlot(r , type_="spatial", histograms=True)
    GeneratorPlot(v , type_="velocity")