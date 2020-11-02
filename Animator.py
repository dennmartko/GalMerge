import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
#matplotlib.use('Agg')
import gc
from matplotlib.pyplot import figure, show, style
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tqdm import tqdm

def remove_axes(ax, hidelabels=True):
    #remove panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    #remove ticks and ticklabels
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_zaxis().set_ticks([])

    #remove spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    #remove grid
    ax.grid(False)
    show()
    #remove labels
    if hidelabels:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    

def plot_linear_cube(ax, midR, L, color=(.224, 1, .078 , 1)):
    x, y, z = midR - L / 2
    dx, dy, dz = [L] * 3

    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': color, 'linewidth' : 0.2}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    return ax

def AnimateOrbit(path, frames, fps=20, sleep=200, window=(-75, 75)):
    style.use('dark_background')
    
    def Frame(i):
        del ax.collections[:]
        del ax.texts[:]
        stars = ax.scatter(xdata[i][:,0].astype(float), xdata[i][:,1].astype(float), xdata[i][:,2].astype(float), s=0.4, color='white')
        ax.text2D(0.75, 1, r"$t = {:.2f}$ Gyr".format(t[i]), fontsize=16, transform=ax.transAxes)
        ax.text2D(0, 1, r"$dt = {}$".format(properties['dt']) + "\n" + r"$\theta = {}$".format(properties['θ']) + "\n" + "Number of Cells: {} \nBodies inside frame: {}".format(properties['NCinFrame'][i],properties['NPinFrame'][i]),
                  ha="left", va="center", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="none", ec="w", pad=0.5))

        print(i)
        return (stars)

    with np.load(path + "/Data.npz",allow_pickle=True) as f:
        xdata = f["r"]

    properties = np.load(path + "/Properties.npz",allow_pickle=True)

    xdata.astype(float)
    t = np.arange(0, properties['dt']*frames, properties['dt'])

    fig = figure(figsize =(10,10))
    ax = fig.add_subplot(111 , projection ='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlim(window)
    ax.set_ylim(window)
    ax.set_zlim(window)
    
    ax.set_xlabel(r"$x$ [kpc]", fontsize=15, labelpad=30)
    ax.set_ylabel(r"$y$ [kpc]", fontsize=15, labelpad=30)
    ax.set_zlabel(r"$z$ [kpc]", fontsize=15, labelpad=30)

    ani = animation.FuncAnimation(fig, Frame, interval=sleep, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animations/animationTEST.mp4"
    ani.save(outfile, fps = fps, writer='ffmpeg', dpi=fig.dpi)

def AnimateCells(path, frames, fps=30, sleep=200, window=(-15, 15)):
    style.use('dark_background')

    def Frame(i):
        ax = fig.add_subplot(111 , projection ='3d')
        remove_axes(ax)
        ax.set(xlim=window, ylim=window, zlim=window)
        #gc.collect()
        del ax.collections[:]
        del ax.texts[:]
        for k, cell in tqdm(enumerate(Cdata[i])):
            if cell.L > 1 and cell.L < 2:
                if k % 4 == 0:
                    ax = plot_linear_cube(ax, cell.midR, cell.L, color=(.224, 1, .078 , 1))
        ax.text2D(0.75, 1, r"$t = {:.2f}$ Gyr".format(t[i]), fontsize=16, transform=ax.transAxes)
        return (ax)

    with np.load(path + "/Cells.npz", allow_pickle=True) as f:
        Cdata = f["cells"]

    properties = np.load(path + "/Properties.npz",allow_pickle=True)
    
    fig = figure(figsize=(10,10))
    t = np.arange(0, properties["dt"] * frames, properties["dt"])

    ani = animation.FuncAnimation(fig, Frame, interval=sleep, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animations/animationCellsTEST.mp4"
    ani.save(outfile, fps = fps, writer='ffmpeg', dpi=fig.dpi)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    AnimateOrbit(path, 10)

    #path = os.path.dirname(os.path.abspath(__file__)) + "/Cells.npz"
    #AnimateCells(path, 100)