import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
#matplotlib.use('Agg')

from matplotlib.pyplot import figure, show, style
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

style.use('dark_background')


def AnimateOrbit(file,frames):
    
    def Frame(i):
        del ax.collections[:]
        stars = ax.scatter(xdata[i][:,0], xdata[i][:,1], list(xdata[i][:,2]), s=0.4, color='white')
        print(i)
        return (stars)


    with np.load(file,allow_pickle=True) as f:
        xdata = f["r"]
    xdata.astype(float)
    length = len(xdata[0][:,0])
    fig = figure(figsize =(10,10))
    ax = fig.add_subplot(111 , projection ='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))
    ax.set_zlim((-20,20))
    ax.set_xlabel(r"$x$ [kpc]", fontsize=15, labelpad=30)
    ax.set_ylabel(r"$y$ [kpc]", fontsize=15, labelpad=30)
    ax.set_zlabel(r"$z$ [kpc]", fontsize=15, labelpad=30)

    ani = animation.FuncAnimation(fig, Frame, interval=200, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animations/animationTEST.mp4"
    ani.save(outfile,fps = 20, writer='ffmpeg',dpi =300)

if __name__ == "__main__":
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
    AnimateOrbit(outfile, 1000)

    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    ax.grid(False)

    x, y, z = np.indices((8, 8, 8))
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube3 = (x < 2) & (y < 2) & (z < 2)
    cube4 = (x < 1) & (y < 1) & (z < 1)
    cube2 = (x < 3) & (y<3) & np.logical_and(3 <= z, z < 6)
    #ax.voxels(np.array([[[0],[0],[0]],[[-1],[2],[3]]]),alpha=0.3)
    setup = dict(edgecolor=(.224, 1, .078 , 1), facecolor=(0,0,0,0))
    ax.voxels(cube1,shade=False, **setup)
    ax.voxels(cube2,shade=False, **setup)
    ax.voxels(cube3,shade=False, **setup)
    ax.voxels(cube4,shade=False, **setup)
    #ax.set(xlabel="x",ylabel="y",zlabel="z")
    plt.show()