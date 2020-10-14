import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from numba import njit


#own imports
import constants as const

def plum_transform(p, Mtot, r0):
    rmax = 10 * r0 #10r0 covers a large region of the potential
    p[:,0] = p[:,0] * rmax ** 3 * Mtot / (4 * r0 ** 3 * np.pi * (1 + rmax ** 2 / r0 ** 2) ** (3 / 2)) #rescale to u such that r= [0,rmax]
    p[:,1] = 2 * np.pi * p[:,1] #rescale to v = phi
    p[:,2] = 2 * p[:,2] - 1 #rescale to w

    p[:,0] = np.sqrt((4 * r0 ** 3 * np.pi * p[:,0]) ** (2 / 3) / (Mtot ** (2 / 3) - (4 * r0 ** 3 * np.pi * p[:,0]) ** (2 / 3) / (r0 ** 2))) + r0 #transform u to r
    p[:,2] = np.arccos(-p[:,2]) #transform w to theta

    #convert to cartesian coordinates
    x = p[:,0] * np.cos(p[:,1]) * np.sin(p[:,2])
    y = p[:,0] * np.sin(p[:,1]) * np.sin(p[:,2])
    z = p[:,0] * np.cos(p[:,2])

    p = np.stack((x, y, z), axis=1)

    return p

#--------------------------------------------------------------------#

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

#function to generate N random particle positions according to the Jaffe density profile
def gen3DJaffe(N, r0):
    f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
    r = r0 * f / (1 - f)
    return np.einsum('i,ij->ij', r, randUnitVec(N))

#function to generate N random particle positions according to the Hernquist density profile
def gen3DHernquist(N, r0):
    f = np.random.rand(N) #fraction of mass M0 enclosed within r, f in [0, 1]
    r = r0 * (f + np.sqrt(f)) / (1 - f)
    return np.einsum('i,ij->ij', r, randUnitVec(N))

# handler function for generating N random particle positions
def generate_r(Npart, r0, Mtot=None, seed=None, type_='plummer'):
    if type_ == "plummer2D":
        if Mtot is None:
            print("Mtot must be defined!")
        p = np.random.rand(Npart, 3)
        p = plum_transform(p, Mtot, r0)
        p = p[:,:2]

    if type_ == "plummer":
        p = gen3DPlummer(Npart, r0)

    if type_ == "jaffe":
        p = gen3DJaffe(Npart, r0)

    if type_ == "hernquist":
        p = gen3DHernquist(Npart, r0)

    return p
    
# Initial velocities for particles in the Plummer model
def vesc_Plummer(r, M, r0):
    return np.sqrt(2 * const.G_ * M / r0) * (1 + r ** 2 / r0 ** 2) ** (-1 / 4)

def vcirc_Plummer(r, M, r0):
    return np.einsum('i,ij->ij', vesc_Plummer(np.linalg.norm(r, axis=1), M, r0) / np.sqrt(2), randUnitVec(len(r)))

# Initial velocities for particles in the Jaffe model
def vesc_Jaffe(r, M, r0):
    return np.sqrt(2 * const.G_ * M / r0)*(-np.log(r / (r0 + r))) ** (1 / 2)

def vcirc_Jaffe(r, M, r0):
    return np.einsum('i,ij->ij', vesc_Jaffe(np.linalg.norm(r, axis=1), M, r0) / np.sqrt(2), randUnitVec(len(r)))

# Initial velocities for particles in the Hernquist model
def vesc_Hernquist(r, M, r0):
    return np.sqrt(2 * const.G_ * M / r0) * (1 + r / r0) ** (-1 / 2)

def vcirc_Hernquist(r, M, r0):
    return np.einsum('i,ij->ij', vesc_Hernquist(np.linalg.norm(r, axis=1), M, r0) / np.sqrt(2), randUnitVec(len(r)))

def generate_v(r, M, r0, type_="plummer"):
    if type_ == "plummer2D":
        #def ve(rr, r0):
        #    return np.sqrt(2 * const.G_ * Mtot / np.sqrt(rr ** 2 + r0 ** 2))

        vesc = np.empty(len(r))
        mag_r = np.linalg.norm(r, axis=1)
        for i in range(len(r)):
            vesc[i] = vesc_Plummer(mag_r[i], M, r0) #compute the escape velocity for each particle
            #print(vesc[i])

        vx = -vesc * r[:,1] / np.sqrt(r[:,0] ** 2 + r[:,1] ** 2)
        vy = vesc * r[:,0] / np.sqrt(r[:,0] ** 2 + r[:,1] ** 2)
        v = np.stack((vx,vy), axis=1)

    if type_ == "plummer":
        v = vcirc_Plummer(r, M, r0)

    if type_ == "jaffe":
        v = vcirc_Jaffe(r, M, r0)

    if type_ == "hernquist":
        v = vcirc_Hernquist(r, M, r0)

    return v

def generate(N, Mtot, r0, disp, type_="plummer"):
    # N: number of particles to generate
    r = generate_r(N, r0=r0, Mtot=Mtot, type_=type_)
    v = generate_v(r, Mtot, r0, type_=type_)
    return r, v

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
    Nparticles = 1000
    θ = 0.6 
    dt = 0.01
    Mtot = 10 ** 9
    r0 = 15
    frames = 400
    disp = 800

    #r, v = generate(Nparticles, Mtot, r0, disp)
    r, v = generate(Nparticles, Mtot, r0, disp, type_="plummer")
    #mag_v = np.linalg.norm(v, axis=1)
    #plt.scatter(range(Nparticles//10),mag_v[::10])
    #plt.show()
    GeneratorPlot(r , type_="spatial", histograms=True)
    GeneratorPlot(v , type_="velocity")