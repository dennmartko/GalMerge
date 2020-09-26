import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#own imports
import constants as const

#module for randomly generating galaxies
def jaffe_transform(p, Mtot, r0):
    rmax = 19 * r0 #95% of the total mass of the galaxy is confined within this region
    p[:,0] = Mtot / (4 * np.pi) * (1 - (1 + rmax / r0) ** (-1)) * p[:,0] - Mtot / (4 * np.pi) #rescale to u
    p[:,1] = 2 * np.pi * p[:,1] #rescale to v = phi
    p[:,2] = 2 * p[:,2] - 1 #rescale to w

    p[:,0] = -r0 * (1 + Mtot / (4 * np.pi * p[:,0])) #transform u to r
    p[:,2] = np.arccos(-p[:,2]) #transform w to theta

    #convert to cartesian coordinates
    x = p[:,0] * np.cos(p[:,1]) * np.sin(p[:,2])
    y = p[:,0] * np.sin(p[:,1]) * np.sin(p[:,2])
    z = p[:,0] * np.cos(p[:,2])

    p = np.stack((x, y, z), axis=1)

    return p


def generate_r(N, type="jaffe"):
    p = np.random.rand(N, 3)
    if type == "jaffe":
        p = jaffe_transform(p, Mtot=10**9*const.G, r0=14*const.kpc2m)*const.m2kpc
    return p
    

def generate_v(N):
    pass

def generate(N):
    # N: number of particles to generate
    r = generate_r(N)
    v = generate_v(N)
    return r, v

def GeneratorPlot(p, type="spatial", histograms=False):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2], s=1, color='white')
    
    if type == "spatial":
        ax.set(xlabel=r"$x$ (kpc)", ylabel=r"$y$ (kpc)", zlabel=r"$z$ (kpc)")
    elif type == "velocity":
        ax.set(xlabel=r"$v_x$ (km/s)", ylabel=r"$v_y$ (km/s)", zlabel=r"$v_z$ (km/s)")

    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    plt.show()

    if histograms and type == "spatial":
        fig = plt.figure(figsize=(10,10))

        histSetup = dict(bins=p.shape[0]//100, color="white", density=True)
        
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
    r = generate_r(100000)
    GeneratorPlot(r, type="spatial", histograms=True)