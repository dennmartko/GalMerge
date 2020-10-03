import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#own imports
import constants as const

# module for randomly generating galaxies
def jaffe_transform(p, Mtot, r0):
    rmax = 99 * r0 #99% #19 * r0 #95% of the total mass of the galaxy is confined within this
                   #region
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


def plum_transform(p, Mtot, r0):
    rmax = 10 * r0 #10r0 covers a large region of the potential
    p[:,0] = p[:,0] * rmax ** 3 * Mtot / (4 * r0 ** 3 * np.pi * (1 + rmax ** 2 / r0 ** 2) ** (3 / 2)) #rescale to u such that r= [0,10r0]
    p[:,1] = 2 * np.pi * p[:,1] #rescale to v = phi
    p[:,2] = 2 * p[:,2] - 1 #rescale to w

    p[:,0] = np.sqrt((4 * r0 ** 3 * np.pi * p[:,0]) ** (2 / 3) / (Mtot ** (2 / 3) - (4 * r0 ** 3 * np.pi * p[:,0]) ** (2 / 3) / (r0 ** 2)))#transform u to r
    p[:,2] = np.arccos(-p[:,2]) #transform w to theta

    #convert to cartesian coordinates
    x = p[:,0] * np.cos(p[:,1]) * np.sin(p[:,2])
    y = p[:,0] * np.sin(p[:,1]) * np.sin(p[:,2])
    z = p[:,0] * np.cos(p[:,2])

    p = np.stack((x, y, z), axis=1)

    return p


def generate_r(N,r0, type="jaffe", Mtot=10 ** 12):
    p = np.random.rand(N, 3)
    if type == "jaffe":
        p = jaffe_transform(p, Mtot=Mtot, r0=r0)
    if type == "plummer":
        p = plum_transform(p, Mtot=Mtot, r0=r0)
    return p
    


def generate_v(N, mag_r, Mtot, disp, r0):
    def ve(M, r, r0):
        if M > 0:
            return np.sqrt(2 * const.G_ * M / np.sqrt(r ** 2 + r0 ** 2))
        else:
            return 0

    vesc = np.empty(N)
    for i in range(N):
        vesc[i] = ve(i * Mtot / N, mag_r[i],r0) #compute the escape velocity for each particle
        #print(vesc[i])
    v = np.random.normal(loc=0, scale=disp, size=(N,3))
    for i in range(N):
        if np.linalg.norm(v[i]) > vesc[i] and i != 0:
            tryagain = True
            while tryagain:
                vtmp = np.random.normal(loc=0, scale=disp, size=3)
                if np.linalg.norm(vtmp) <= vesc[i]:
                    v[i] = vtmp
                    tryagain = False
        elif i == 0:
            v[i] = np.zeros(3)

    return v


def generate(N, Mtot=10 ** 12, r0=10, disp=10):
    # N: number of particles to generate
    r = generate_r(N,r0=r0, type="plummer", Mtot=Mtot)
    mag_r = np.linalg.norm(r, axis=1) #size of r
    indices = np.argsort(mag_r)
    r = r[indices,:] #sort r according to the size of each position vector
    mag_r = mag_r[indices] #sort the r size array
    v = generate_v(N, mag_r, Mtot, disp,r0=r0) + np.array([40,150,0])
    return r, v

def GeneratorPlot(p, type="spatial", histograms=False):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2], s=1, color='white')
    
    if type == "spatial":
        ax.set(xlabel=r"$x$ (kpc)", ylabel=r"$y$ (kpc)", zlabel=r"$z$ (kpc)")
    elif type == "velocity":
        ax.set(xlabel=r"$v_x$ (kpc/gyr)", ylabel=r"$v_y$ (kpc/gyr)", zlabel=r"$v_z$ (kpc/gyr)")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.show()

    if histograms and type == "spatial":
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
    r, v = generate(1000)
    mag_v = np.linalg.norm(v, axis=1)
    plt.scatter(range(100),mag_v[::10])
    plt.show()
    GeneratorPlot(r , type="spatial", histograms=True)
    GeneratorPlot(v , type="velocity")