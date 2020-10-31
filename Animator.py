import numpy as np
import matplotlib
import os

#matplotlib.use('Agg')

from matplotlib.pyplot import figure, show, style
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

style.use('dark_background')
import numba
from numba import jit


def AnimateOrbit(file,frames):
    
    def Frame(i):
        del ax.collections[:]
        stars = ax.scatter(xdata[i][:,0],xdata[i][:,1],s=0.4,color="white")
        return stars


    with np.load(file,allow_pickle=True) as f:
        xdata = f["r"]

    fig = figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.set_xlim((-100,100))
    ax.set_ylim((-100,100))
    ax.set_xlabel(r"$x$ [kpc]", fontsize=15, labelpad=30)
    ax.set_ylabel(r"$y$ [kpc]", fontsize=15, labelpad=30)

    ani = animation.FuncAnimation(fig, Frame, interval=200, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animationTEST.mp4"
    writer = animation.writers['ffmpeg']
    writer = writer(fps=12)
    ani.save(outfile, writer=writer, dpi = 300)


#outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
#AnimateOrbit(outfile, 100)